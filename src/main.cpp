
#include <optix.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include "OpenImageIO\imageio.h"
#include "OpenImageIO\imagebuf.h"
#include <stdio.h>
#include <exception>

#include <time.h>
#include <thread>
#include <chrono>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <winternl.h>
// The cfgmgr32 header is necessary for interrogating driver information in the registry.
#include <cfgmgr32.h>
#else
#include <dlfcn.h>
#endif

#define DENOISER_MAJOR_VERSION 3
#define DENOISER_MINOR_VERSION 0

// Our global image handles
OIIO::ImageBuf* g_input_beauty = nullptr;
OIIO::ImageBuf* g_input_albedo = nullptr;
OIIO::ImageBuf* g_input_normal = nullptr;

// Logging verbosity level
int g_verbosity = 2;

// Application start time
std::chrono::high_resolution_clock::time_point g_app_start_time;

// Device count
std::vector<cudaDeviceProp> g_device_props;

// OptiX function table
OptixFunctionTable g_api;

void cleanup()
{
    if (g_input_beauty) delete g_input_beauty;
    if (g_input_albedo) delete g_input_albedo;
    if (g_input_normal) delete g_input_normal;
}

std::string getTime()
{
    std::chrono::duration<double, std::milli> time_span = std::chrono::high_resolution_clock::now() - g_app_start_time;
    double milliseconds = time_span.count();
    int seconds = floor(milliseconds / 1000.0);
    int minutes = floor((float(seconds) / 60.f));
    milliseconds -= seconds * 1000.0;
    seconds -= minutes * 60;
    char s[9];
    sprintf(s, "%02d:%02d:%03d", minutes, seconds, (int)milliseconds);
    return std::string(s);
}

template<typename... Args>
void PrintInfo(char *c, Args... args)
{
    if (!g_verbosity)
        return;
    char buffer[256];
    sprintf(buffer, c, args...);
    std::cout<<getTime()<<"       | "<<buffer<<std::endl;
}

template<typename... Args>
void PrintError(char *c, Args... args)
{
    char buffer[256];
    sprintf(buffer, c, args...);
    std::cerr<<getTime()<<" ERROR | "<<buffer<<std::endl;
}

#ifdef _WIN32
int getSysOpType()
{
    int ret = 0;
    NTSTATUS(WINAPI *RtlGetVersion)(LPOSVERSIONINFOEXW);
    OSVERSIONINFOEXW osInfo;

    *(FARPROC*)&RtlGetVersion = GetProcAddress(GetModuleHandleA("ntdll"), "RtlGetVersion");

    if (NULL != RtlGetVersion)
    {
        osInfo.dwOSVersionInfoSize = sizeof(osInfo);
        RtlGetVersion(&osInfo);
        ret = osInfo.dwMajorVersion;
    }
    return ret;
}
#endif

void exitfunc(int exit_code)
{
#ifdef _WIN32
    if (getSysOpType() < 10)
    {
        HANDLE tmpHandle = OpenProcess(PROCESS_ALL_ACCESS, TRUE, GetCurrentProcessId());
        if (tmpHandle != NULL)
        {
            PrintInfo("terminating...");
            std::this_thread::sleep_for(std::chrono::seconds(1)); // delay 1s
            TerminateProcess(tmpHandle, 0);
        }
    }
#endif
	exit(exit_code);
}

inline void cudaCheckReportError(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        PrintError("CUDA error at %s:%d code=%d(%s) \"%s\"", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        cleanup();
        exitfunc(EXIT_FAILURE);
    }
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define CU_CHECK(val) cudaCheckReportError((val), #val, __FILE__, __LINE__)

inline void optixCheckReportError(OptixResult result, char const *const func, const char *const file, int const line)
{
    if (result != OPTIX_SUCCESS)
    {
        PrintError("OptiX error at %s:%d code=%d(%s) \"%s\"", file, line, static_cast<unsigned int>(result), g_api.optixGetErrorName(result), func);
        cleanup();
        exitfunc(EXIT_FAILURE);
    }
}

// This will output the proper OptiX error strings in the event
// that a OptiX host call returns an error
#define OPTIX_CHECK(val) optixCheckReportError((val), #val, __FILE__, __LINE__)

// The following windows optix loading utility was taken from Nvidia's OptiX
// examples at https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/intro_denoiser/src/Application.cpp
#ifdef _WIN32
// Code based on helper function in optix_stubs.h
static void* optixLoadWindowsDll(void)
{
    const char* optixDllName = "nvoptix.dll";
    void* handle = NULL;

    // Get the size of the path first, then allocate
    unsigned int size = GetSystemDirectoryA(NULL, 0);
    if (size == 0)
    {
        // Couldn't get the system path size, so bail
        return NULL;
    }

    size_t pathSize = size + 1 + strlen(optixDllName);
    char*  systemPath = (char*) malloc(pathSize);

    if (GetSystemDirectoryA(systemPath, size) != size - 1)
    {
        // Something went wrong
        free(systemPath);
        return NULL;
    }

    strcat(systemPath, "\\");
    strcat(systemPath, optixDllName);

    handle = LoadLibraryA(systemPath);

    free(systemPath);

    if (handle)
    {
        return handle;
    }

    // If we didn't find it, go looking in the register store.  Since nvoptix.dll doesn't
    // have its own registry entry, we are going to look for the OpenGL driver which lives
    // next to nvoptix.dll. 0 (null) will be returned if any errors occured.

    static const char* deviceInstanceIdentifiersGUID = "{4d36e968-e325-11ce-bfc1-08002be10318}";
    const ULONG        flags = CM_GETIDLIST_FILTER_CLASS | CM_GETIDLIST_FILTER_PRESENT;
    ULONG              deviceListSize = 0;

    if (CM_Get_Device_ID_List_SizeA(&deviceListSize, deviceInstanceIdentifiersGUID, flags) != CR_SUCCESS)
    {
        return NULL;
    }

    char* deviceNames = (char*) malloc(deviceListSize);

    if (CM_Get_Device_ID_ListA(deviceInstanceIdentifiersGUID, deviceNames, deviceListSize, flags))
    {
        free(deviceNames);
        return NULL;
    }

    DEVINST devID = 0;

    // Continue to the next device if errors are encountered.
    for (char* deviceName = deviceNames; *deviceName; deviceName += strlen(deviceName) + 1)
    {
      if (CM_Locate_DevNodeA(&devID, deviceName, CM_LOCATE_DEVNODE_NORMAL) != CR_SUCCESS)
      {
          continue;
      }

      HKEY regKey = 0;
      if (CM_Open_DevNode_Key(devID, KEY_QUERY_VALUE, 0, RegDisposition_OpenExisting, &regKey, CM_REGISTRY_SOFTWARE) != CR_SUCCESS)
      {
          continue;
      }

      const char* valueName = "OpenGLDriverName";
      DWORD       valueSize = 0;

      LSTATUS     ret = RegQueryValueExA(regKey, valueName, NULL, NULL, NULL, &valueSize);
      if (ret != ERROR_SUCCESS)
      {
          RegCloseKey(regKey);
          continue;
      }

      char* regValue = (char*) malloc(valueSize);
      ret = RegQueryValueExA(regKey, valueName, NULL, NULL, (LPBYTE) regValue, &valueSize);
      if (ret != ERROR_SUCCESS)
      {
          free(regValue);
          RegCloseKey(regKey);
          continue;
      }

      // Strip the OpenGL driver dll name from the string then create a new string with
      // the path and the nvoptix.dll name
      for (int i = valueSize - 1; i >= 0 && regValue[i] != '\\'; --i)
      {
          regValue[i] = '\0';
      }

      size_t newPathSize = strlen(regValue) + strlen(optixDllName) + 1;
      char*  dllPath = (char*) malloc(newPathSize);
      strcpy(dllPath, regValue);
      strcat(dllPath, optixDllName);

      free(regValue);
      RegCloseKey(regKey);

      handle = LoadLibraryA((LPCSTR) dllPath);
      free(dllPath);

      if (handle)
      {
          break;
      }
    }

    free(deviceNames);

    return handle;
}
#endif

// The following windows optix function table loading utility was taken from Nvidia's OptiX
// examples at https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/intro_denoiser/src/Application.cpp
OptixResult initOptiXFunctionTable()
{
#ifdef _WIN32
    void* handle = optixLoadWindowsDll();
    if (!handle)
    {
        return OPTIX_ERROR_LIBRARY_NOT_FOUND;
    }

    void* symbol = reinterpret_cast<void*>(GetProcAddress((HMODULE) handle, "optixQueryFunctionTable"));
    if (!symbol)
    {
        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
    }
#else
    void* handle = dlopen("libnvoptix.so.1", RTLD_NOW);
    if (!handle)
    {
        return OPTIX_ERROR_LIBRARY_NOT_FOUND;
    }

    void* symbol = dlsym(handle, "optixQueryFunctionTable");
    if (!symbol)
    {
        return OPTIX_ERROR_ENTRY_SYMBOL_NOT_FOUND;
    }
#endif

    OptixQueryFunctionTable_t* optixQueryFunctionTable = reinterpret_cast<OptixQueryFunctionTable_t*>(symbol);

    return optixQueryFunctionTable(OPTIX_ABI_VERSION, 0, 0, 0, &g_api, sizeof(OptixFunctionTable));
}

void printParams()
{
    // Always print parameters if needed
    int old_verbosity = g_verbosity;
    g_verbosity = 1;
    PrintInfo("Command line parameters");
    PrintInfo("-v [int]      : log verbosity level 0:disabled 1:simple 2:full (default 2)");
    PrintInfo("-i [string]   : path to input image");
    PrintInfo("-o [string]   : path to output image");
    PrintInfo("-a [string]   : path to input albedo AOV (optional)");
    PrintInfo("-n [string]   : path to input normal AOV (optional, requires albedo AOV)");
    PrintInfo("-b [float]    : blend amount (default 0)");
    PrintInfo("-hdr [int]    : Use HDR training data (default 1)");
    PrintInfo("-gpu [int]    : Select which GPU to use for denoising (default 0)");
    PrintInfo("-repeat [int] : Execute the denoiser N times. Useful for profiling.");
    g_verbosity = old_verbosity;
}

bool DiscoverDevices()
{
    //Lets test some cuda stuff
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count))
    {
        PrintError("Failed to get device information");
        return false;
    }
    PrintInfo("Found %d CUDA device(s)", device_count);
    if(device_count == 0){
        PrintError("No Nvidia GPUs found");
        return false;
    }
    for (int i=0; i < device_count; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        PrintInfo("GPU %d: %s (compute %d.%d) with %dMB memory", i, prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1024 / 1024);
        g_device_props.push_back(prop);
    }
    return true;
}

int main(int argc, char *argv[])
{
    g_app_start_time = std::chrono::high_resolution_clock::now();
    if (argc > 1)
    {
        for (int i=1; i<argc; i++)
        {
            if (strcmp(argv[i], "-v"))
                continue;
            i++;
            if (i >= argc)
            {
               PrintError("incorrect number of arguments for flag -v");
            }
            g_verbosity = std::stoi(std::string ( argv[i] ));
            break;
        }
    }
    PrintInfo("Launching Nvidia AI Denoiser command line app v%d.%d", DENOISER_MAJOR_VERSION, DENOISER_MINOR_VERSION);
    PrintInfo("Created by Declan Russell (25/12/2017 ~ Merry Christmas!)");

    if (!DiscoverDevices())
        exitfunc(EXIT_FAILURE);

    OptixResult result = initOptiXFunctionTable();
    if (result != OPTIX_SUCCESS)
    {
        // TODO: It would be nice to get the actual error string here, but how do
        //       we get that if we can't dlopen OptiX?
        PrintError("Cannot initialize OptiX library (%d)", result);
        exitfunc(EXIT_FAILURE);
    }

    bool b_loaded, n_loaded, a_loaded;
    b_loaded = n_loaded = a_loaded = false;

    // Pass our command line args
    std::string out_path;
    float blend = 0.f;
    unsigned int hdr = 1;
    unsigned int selected_device_id = 0;
    unsigned int num_runs = 1;
    if (argc == 1)
    {
        printParams();
        exitfunc(EXIT_SUCCESS);
    }
    for (int i=1; i<argc; i++)
    {
        const std::string arg( argv[i] );
        if (arg == "-i")
        {
            i++;
            std::string path( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Input image: %s", path.c_str());
            g_input_beauty = new OIIO::ImageBuf(path);
            if (g_input_beauty->init_spec(path, 0, 0))
            {
                if (g_verbosity >= 2)
                    PrintInfo("Loaded successfully");
                b_loaded = true;
            }
            else
            {
                PrintError("Failed to load input image");
                PrintError("[OIIO]: %s", g_input_beauty->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if (arg == "-n")
        {
            i++;
            std::string path( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Normal image: %s", path.c_str());
            g_input_normal = new OIIO::ImageBuf(path);
            if (g_input_normal->init_spec(path, 0, 0))
            {
                if (g_verbosity >= 2)
                    PrintInfo("Loaded successfully");
                n_loaded = true;
            }
            else
            {
                PrintError("Failed to load normal image");
                PrintError("[OIIO]: %s", g_input_normal->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if (arg == "-a")
        {
            i++;
            std::string path( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Albedo image: %s", path.c_str());
            g_input_albedo = new OIIO::ImageBuf(path);
            if (g_input_albedo->init_spec(path, 0, 0))
            {
                if (g_verbosity >= 2)
                    PrintInfo("Loaded successfully");
                a_loaded = true;
            }
            else
            {
                PrintError("Failed to load albedo image");
                PrintError("[OIIO]: %s", g_input_albedo->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if(arg == "-o")
        {
            i++;
            out_path = std::string( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Output image: %s", out_path.c_str());
        }
        else if (arg == "-b")
        {
            i++;
            std::string blend_string( argv[i] );
            blend = std::stof(blend_string);
            if (g_verbosity >= 2)
                PrintInfo("Blend amount: %f", blend);
        }
        else if (arg == "-hdr")
        {
            i++;
            std::string hdr_string( argv[i] );
            hdr = std::stoi(hdr_string);
            if (g_verbosity >= 2)
                PrintInfo((hdr) ? "HDR training data enabled" : "HDR training data disabled");
        }
        else if (arg == "-gpu")
        {
            i++;
            std::string gpu_string( argv[i] );
            int gpu = std::stoi(gpu_string);
            if (gpu < 0 || gpu >= (int)g_device_props.size())
            {
                PrintError("GPU ID %d does not match any known GPU", gpu);
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
            selected_device_id = (unsigned int)gpu;
        }
        else if (arg == "-repeat")
        {
            i++;
            std::string repeat_string( argv[i] );
            num_runs = std::max(std::stoi(repeat_string), 1);
            if (g_verbosity >= 2)
                PrintInfo("Number of repeats set to %d", num_runs);
        }
        else if (arg == "-h" || arg == "--help")
        {
            printParams();
        }
    }

    if (g_verbosity >= 2)
        PrintInfo("Using GPU %d: %s (compute %d.%d) with %dMB memory", selected_device_id, g_device_props[selected_device_id].name,
                                                                    g_device_props[selected_device_id].major, g_device_props[selected_device_id].minor,
                                                                    g_device_props[selected_device_id].totalGlobalMem / 1024 / 1024);

    // Check if a beauty has been loaded
    if (!b_loaded)
    {
        PrintError("No input image could be loaded");
        cleanup();
        exitfunc(EXIT_FAILURE);
    }

    // If a normal AOV is loaded then we also require an albedo AOV
    if (n_loaded && !a_loaded)
    {
        PrintError("You cannot use a normal AOV without an albedo");
        cleanup();
        exitfunc(EXIT_FAILURE);
    }

    // Check for a file extension
    int x = (int)out_path.find_last_of(".");
    x++;
    const char* ext_c = out_path.c_str()+x;
    std::string ext(ext_c);
    if (!ext.size())
    {
        PrintError("No output file extension");
        cleanup();
        exitfunc(EXIT_FAILURE);
    }

    OIIO::ROI beauty_roi, albedo_roi, normal_roi;
    beauty_roi = OIIO::get_roi_full(g_input_beauty->spec());
    int b_width = beauty_roi.width();
    int b_height = beauty_roi.height();
    if (a_loaded)
    {
        albedo_roi = OIIO::get_roi_full(g_input_albedo->spec());
        if (n_loaded)
            normal_roi = OIIO::get_roi_full(g_input_normal->spec());
    }

    // Check that our feature buffers are the same resolution as our beauty
    int a_width = (a_loaded) ? albedo_roi.width() : 0;
    int a_height = (a_loaded) ? albedo_roi.height() : 0;
    if (a_loaded)
    {
        if (a_width != b_width || a_height != b_height)
        {
            PrintError("Aldedo image not same resolution as beauty");
            cleanup();
            exitfunc(EXIT_FAILURE);
        }
    }

    int n_width = (n_loaded) ? normal_roi.width() : 0;
    int n_height = (n_loaded) ? normal_roi.height() : 0;
    if (n_loaded)
    {
        if (n_width != b_width || n_height != b_height)
        {
            PrintError("Normal image not same resolution as beauty");
            cleanup();
            exitfunc(EXIT_FAILURE);
        }
    }

    // Get our pixel data
    std::vector<float> beauty_pixels(b_width * b_height * beauty_roi.nchannels());
    g_input_beauty->get_pixels(beauty_roi, OIIO::TypeDesc::FLOAT, &beauty_pixels[0]);

    // Catch optix exceptions
    try
    {
        // Select the GPU we want to use
        CU_CHECK(cudaSetDevice(selected_device_id));

        // The runtime API lazily initializes its CUDA context on first usage
        // Calling cudaFree here forces our context to initialize
        // TODO: Time this?
        CU_CHECK(cudaFree(0));

        // Create a stream to run the denoiser on
        cudaStream_t cuda_stream;
        CU_CHECK(cudaStreamCreate(&cuda_stream));

        // Initialize our optix context
        CUcontext cuCtx = 0; // Zero means take the current context
        OptixDeviceContext optix_context = nullptr;
        result = g_api.optixDeviceContextCreate(cuCtx, nullptr, &optix_context);
        if (result != OPTIX_SUCCESS)
        {
            PrintError("Could not create OptiX context: (%d) %s", result, g_api.optixGetErrorName(result));
            cleanup();
            exitfunc(EXIT_FAILURE);
        }

        // Set the denoiser options
        OptixDenoiserOptions denoiser_options = {};
        denoiser_options.guideAlbedo = a_loaded;
        denoiser_options.guideNormal = n_loaded;

        // Iniitalize the OptiX denoiser
        OptixDenoiser optix_denoiser = nullptr;
        OptixDenoiserModelKind model = (hdr) ? OPTIX_DENOISER_MODEL_KIND_HDR : OPTIX_DENOISER_MODEL_KIND_LDR;
        OPTIX_CHECK( g_api.optixDenoiserCreate(optix_context, model, &denoiser_options, &optix_denoiser) );
        // Compute memory needed for the denoiser to exist on the GPU
        OptixDenoiserSizes denoiser_sizes;
        memset(&denoiser_sizes, 0, sizeof(OptixDenoiserSizes));
        OPTIX_CHECK( g_api.optixDenoiserComputeMemoryResources(optix_denoiser, b_width, b_height, &denoiser_sizes) );
        // Allocate this space on the GPu
        void* denoiser_state_buffer = nullptr;
        void* denoiser_scratch_buffer = nullptr;
        CU_CHECK(cudaMalloc(&denoiser_state_buffer, denoiser_sizes.stateSizeInBytes));
        CU_CHECK(cudaMalloc(&denoiser_scratch_buffer, denoiser_sizes.withoutOverlapScratchSizeInBytes));
        // Setup the denoiser
        OPTIX_CHECK( g_api.optixDenoiserSetup(optix_denoiser, cuda_stream,
                                              b_width, b_height,
                                              (CUdeviceptr)denoiser_state_buffer,   denoiser_sizes.stateSizeInBytes,
                                              (CUdeviceptr)denoiser_scratch_buffer, denoiser_sizes.withoutOverlapScratchSizeInBytes) );

        // Set the denoiser parameters
        OptixDenoiserParams denoiser_params = {};
        // TODO: Expose option for this
        denoiser_params.denoiseAlpha = 0;
        denoiser_params.blendFactor = blend;
        CU_CHECK(cudaMalloc((void**)&denoiser_params.hdrIntensity, sizeof(float)));

        // Create our GPU buffers and load our image data into them
        void* beauty_buffer = nullptr;
        void* albedo_buffer = nullptr;
        void* normal_buffer = nullptr;
        void* out_buffer    = nullptr;
        CU_CHECK(cudaMalloc(&beauty_buffer, sizeof(float) * 4 * b_width * b_height));
        CU_CHECK(cudaMalloc(&out_buffer   , sizeof(float) * 4 * b_width * b_height));
        if (a_loaded)
            CU_CHECK(cudaMalloc(&albedo_buffer, sizeof(float) * 4 * a_width * a_height));
        if (n_loaded)
            CU_CHECK(cudaMalloc(&normal_buffer, sizeof(float) * 4 * n_width * n_height));

        unsigned int buffer_size = 4 * b_width * b_height;
        std::vector<float> host_scratch(buffer_size, 0.f);
        float* ptr = (float*)&host_scratch[0];
        unsigned int pixel_idx = 0;
        for(unsigned int y=0; y<b_height; y++)
        for(unsigned int x=0; x<b_width; x++)
        {
            memcpy(ptr, &beauty_pixels[pixel_idx], sizeof(float) * beauty_roi.nchannels());
            ptr += 4;
            pixel_idx += beauty_roi.nchannels();
        }
        // Copy our data to the GPU
        CU_CHECK(cudaMemcpy(beauty_buffer, &host_scratch[0], sizeof(float) * buffer_size, cudaMemcpyHostToDevice));
        ptr = nullptr;

        if (a_loaded)
        {
            std::vector<float> albedo_pixels(a_width * a_height * albedo_roi.nchannels());
            g_input_albedo->get_pixels(albedo_roi, OIIO::TypeDesc::FLOAT, &albedo_pixels[0]);
            memset(&host_scratch[0], 0, sizeof(float) * buffer_size);
            ptr = (float*)&host_scratch[0];
            pixel_idx = 0;
            for(unsigned int y=0; y<a_height; y++)
            for(unsigned int x=0; x<a_width; x++)
            {
                memcpy(ptr, &albedo_pixels[pixel_idx], sizeof(float) * albedo_roi.nchannels());
                ptr += 4;
                pixel_idx += albedo_roi.nchannels();
            }
            // Copy our data to the GPU
            CU_CHECK(cudaMemcpy(albedo_buffer, &host_scratch[0], sizeof(float) * buffer_size, cudaMemcpyHostToDevice));
            ptr = nullptr;
        }

        if (n_loaded)
        {
            std::vector<float> normal_pixels(n_width * n_height * normal_roi.nchannels());
            g_input_normal->get_pixels(normal_roi, OIIO::TypeDesc::FLOAT, &normal_pixels[0]);
            memset(&host_scratch[0], 0, sizeof(float) * buffer_size);
            ptr = (float*)&host_scratch[0];
            pixel_idx = 0;
            for(unsigned int y=0; y<n_height; y++)
            for(unsigned int x=0; x<n_width; x++)
            {
                memcpy(ptr, &normal_pixels[pixel_idx], sizeof(float) * normal_roi.nchannels());
                ptr += 4;
                pixel_idx += normal_roi.nchannels();
            }
            // Copy our data to the GPU
            CU_CHECK(cudaMemcpy(normal_buffer, &host_scratch[0], sizeof(float) * buffer_size, cudaMemcpyHostToDevice));
            ptr = nullptr;
        }

        // Create and set our OptiX layers
        OptixDenoiserLayer layer       = {};
        // Input
        layer.input.data               = (CUdeviceptr)beauty_buffer;
        layer.input.width              = b_width;
        layer.input.height             = b_height;
        layer.input.rowStrideInBytes   = b_width * sizeof(float) * 4;
        layer.input.pixelStrideInBytes = sizeof(float) * 4;
        layer.input.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        // Output
        layer.output.data               = (CUdeviceptr)out_buffer;
        layer.output.width              = b_width;
        layer.output.height             = b_height;
        layer.output.rowStrideInBytes   = b_width * sizeof(float) * 4;
        layer.output.pixelStrideInBytes = sizeof(float) * 4;
        layer.output.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixDenoiserGuideLayer guide_layer = {};
        // albedo
        if (a_loaded)
        {
            guide_layer.albedo.data               = (CUdeviceptr)albedo_buffer;
            guide_layer.albedo.width              = a_width;
            guide_layer.albedo.height             = a_height;
            guide_layer.albedo.rowStrideInBytes   = a_width * sizeof(float) * 4;
            guide_layer.albedo.pixelStrideInBytes = sizeof(float) * 4;
            guide_layer.albedo.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
        }

        // normal
        if (n_loaded)
        {
            guide_layer.normal.data               = (CUdeviceptr)normal_buffer;
            guide_layer.normal.width              = n_width;
            guide_layer.normal.height             = n_height;
            guide_layer.normal.rowStrideInBytes   = n_width * sizeof(float) * 4;
            guide_layer.normal.pixelStrideInBytes = sizeof(float) * 4;
            guide_layer.normal.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
        }

        // Execute denoise
        int sum = 0;
        for (unsigned int i = 0; i < num_runs; i++)
        {
            PrintInfo("Denoising...");
            clock_t start = clock(), diff;
            // Compute the intensity of the input image
            OPTIX_CHECK( g_api.optixDenoiserComputeIntensity(optix_denoiser, cuda_stream, &layer.input, denoiser_params.hdrIntensity,
                                                            (CUdeviceptr)denoiser_scratch_buffer, denoiser_sizes.withoutOverlapScratchSizeInBytes) );

            // Execute the denoiser
            OPTIX_CHECK( g_api.optixDenoiserInvoke(optix_denoiser, cuda_stream, &denoiser_params,
                                                (CUdeviceptr)denoiser_state_buffer, denoiser_sizes.stateSizeInBytes,
                                                &guide_layer, &layer, 1 /*num input layers*/, 0, 0,
                                                (CUdeviceptr)denoiser_scratch_buffer, denoiser_sizes.withoutOverlapScratchSizeInBytes) );
            diff = clock() - start;
            int msec = diff * 1000 / CLOCKS_PER_SEC;
            if (num_runs > 1)
                PrintInfo("Denoising run %d complete in %d.%03d seconds", i, msec/1000, msec%1000);
            else
                PrintInfo("Denoising complete in %d.%03d seconds", msec/1000, msec%1000);
            sum += msec;
        }
        if (num_runs > 1)
        {
            sum /= num_runs;
            PrintInfo("Denoising avg of %d complete in %d.%03d seconds", num_runs, sum/1000, sum%1000);
        }

        // Copy denoised image back to the cpu
        CU_CHECK(cudaMemcpy(&host_scratch[0], out_buffer, sizeof(float) * buffer_size, cudaMemcpyDeviceToHost));
        ptr = (float*)&host_scratch[0];
        pixel_idx = 0;
        for(unsigned int y=0; y<b_height; y++)
        for(unsigned int x=0; x<b_width; x++)
        {
            memcpy(&beauty_pixels[pixel_idx], ptr, sizeof(float) * beauty_roi.nchannels());
            ptr += 4;
            pixel_idx += beauty_roi.nchannels();
        }
        ptr = nullptr;

        // Remove our gpu buffers
        CU_CHECK(cudaFree((void*)denoiser_params.hdrIntensity));
        CU_CHECK(cudaFree(beauty_buffer));
        CU_CHECK(cudaFree(albedo_buffer));
        CU_CHECK(cudaFree(normal_buffer));
        CU_CHECK(cudaFree(out_buffer   ));
        // Destroy the denoiser
        CU_CHECK(cudaFree(denoiser_state_buffer));
        CU_CHECK(cudaFree(denoiser_scratch_buffer));
        OPTIX_CHECK( g_api.optixDenoiserDestroy(optix_denoiser) );
        // Destroy the OptiX context
        OPTIX_CHECK( g_api.optixDeviceContextDestroy(optix_context) );
        // Delete our CUDA stream as well
        CU_CHECK(cudaStreamDestroy(cuda_stream));

    }
    catch (const std::exception &e)
    {
        PrintError("[OptiX]: %s", e.what());
        cleanup();
        exitfunc(EXIT_FAILURE);
    }


    // If the image already exists delete it
    remove(out_path.c_str());

    // Set our OIIO pixels
    if (!g_input_beauty->set_pixels(beauty_roi, OIIO::TypeDesc::FLOAT, &beauty_pixels[0]))
        PrintError("Something went wrong setting pixels");

    // Save the output image
    PrintInfo("Saving to: %s", out_path.c_str());
    if (g_input_beauty->write(out_path))
        PrintInfo("Done!");
    else
    {
        PrintError("Could not save file %s", out_path.c_str());
        PrintError("[OIIO]: %s", g_input_beauty->geterror().c_str());
    }

    cleanup();
    exitfunc(EXIT_SUCCESS);
}

