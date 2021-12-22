
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
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
#include <string>
#include <unordered_map>
#ifdef _WIN32
#include <windows.h>
#include <winternl.h>
#endif

#define DENOISER_MAJOR_VERSION 3
#define DENOISER_MINOR_VERSION 0

// Simple struct to hold information about our images
struct ImageInfo
{
    std::string filename;
    std::string output_filename;
    OIIO::ImageBuf* data;
};

// Our global image handles
ImageInfo g_input_beauty;
ImageInfo g_input_albedo;
ImageInfo g_input_normal;
std::unordered_map<int, ImageInfo> g_input_aov;

// Logging verbosity level
int g_verbosity = 2;

// Application start time
std::chrono::high_resolution_clock::time_point g_app_start_time;

// Device count
std::vector<cudaDeviceProp> g_device_props;

void cleanup()
{
    if (g_input_beauty.data) delete g_input_beauty.data;
    if (g_input_albedo.data) delete g_input_albedo.data;
    if (g_input_normal.data) delete g_input_normal.data;
    for (auto& i: g_input_aov)
        if (i.second.data)
            delete i.second.data;
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
    std::cout<<getTime()<<"         | "<<buffer<<std::endl;
}

template<typename... Args>
void PrintWarning(char *c, Args... args)
{
    char buffer[256];
    sprintf(buffer, c, args...);
    std::cerr<<getTime()<<" WARNING | "<<buffer<<std::endl;
}

template<typename... Args>
void PrintError(char *c, Args... args)
{
    char buffer[256];
    sprintf(buffer, c, args...);
    std::cerr<<getTime()<<" ERROR   | "<<buffer<<std::endl;
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

inline void imageConvertFormat(float* in_ptr, uint8_t in_size, float* out_ptr, uint8_t out_size, unsigned int width, unsigned int height)
{
    for (unsigned int y=0; y<height; y++)
    for (unsigned int x=0; x<width; x++)
    {
        memcpy(out_ptr, in_ptr, sizeof(float) * in_size);
        out_ptr += out_size;
        in_ptr += in_size;
    }
}

void printParams()
{
    // Always print parameters if needed
    int old_verbosity = g_verbosity;
    g_verbosity = 1;
    PrintInfo("Command line parameters");
    PrintInfo("-v [int]         : log verbosity level 0:disabled 1:simple 2:full (default 2)");
    PrintInfo("-i [string]      : path to input image");
    PrintInfo("-aov%d [string]  : path to additional input AOV image to denoise");
    PrintInfo("-oaov%d [string] : path to additional AOV output image to denoise");
    PrintInfo("-o [string]      : path to output image");
    PrintInfo("-os [string]     : output suffix appended to input filename to create output image filename");
    PrintInfo("-a [string]      : path to input albedo AOV (optional)");
    PrintInfo("-n [string]      : path to input normal AOV (optional, requires albedo AOV)");
    PrintInfo("-b [float]       : blend amount (default 0)");
    PrintInfo("-hdr [int]       : Use HDR training data (default 1)");
    PrintInfo("-gpu [int]       : Select which GPU to use for denoising (default 0)");
    PrintInfo("-repeat [int]    : Execute the denoiser N times. Useful for profiling.");
    g_verbosity = old_verbosity;
}

bool discoverDevices()
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

    if (!discoverDevices())
        exitfunc(EXIT_FAILURE);

    OptixResult result = optixInit();
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
    std::string out_suffix;
    float blend = 0.f;
    unsigned int hdr = 1;
    unsigned int selected_device_id = 0;
    unsigned int num_runs = 1;
    bool denoise_aovs = false;
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
            g_input_beauty.filename = std::string( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Input image: %s", g_input_beauty.filename.c_str());
            g_input_beauty.data = new OIIO::ImageBuf(g_input_beauty.filename);
            if (g_input_beauty.data->init_spec(g_input_beauty.filename, 0, 0))
            {
                if (g_verbosity >= 2)
                    PrintInfo("Loaded successfully");
                b_loaded = true;
            }
            else
            {
                PrintError("Failed to load input image");
                PrintError("[OIIO]: %s", g_input_beauty.data->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if (arg.find("-aov") != std::string::npos)
        {
            if (arg.size() == 4)
            {
                PrintError("-aov parameter requires number id such as -aov0");
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
            int aov_id = std::stoi(arg.substr(4, arg.size()));
            i++;
            ImageInfo info;
            info.filename = std::string( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Additional AOV %d input: %s", aov_id, info.filename.c_str());
            info.data = new OIIO::ImageBuf(info.filename);
            if (info.data->init_spec(info.filename, 0, 0))
            {
                if (g_verbosity >= 2)
                    PrintInfo("Loaded successfully");
                b_loaded = true;
                auto it = g_input_aov.find(aov_id);
                if (it == g_input_aov.end())
                {
                    g_input_aov[aov_id] = info;
                }
                else
                {
                    it->second.filename = info.filename;
                    if (it->second.data)
                        delete it->second.data;
                    it->second.data = info.data;
                }
                denoise_aovs = true;
            }
            else
            {
                PrintError("Failed to AOV input image");
                PrintError("[OIIO]: %s", info.data->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if (arg.find("-oaov") != std::string::npos)
        {
            if (arg.size() == 5)
            {
                PrintError("-oaov parameter requires number id such as -oaov0");
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
            int aov_id = std::stoi(arg.substr(5, arg.size()));

            i++;
            ImageInfo info;
            info.output_filename = std::string( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Additional AOV %d output: %s", aov_id, info.output_filename.c_str());

            auto it = g_input_aov.find(aov_id);
            if (it == g_input_aov.end())
            {
                g_input_aov[aov_id] = info;
            }
            else
            {
                it->second.output_filename = info.output_filename;
            }
        }
        else if (arg == "-n")
        {
            i++;
            g_input_normal.filename = std::string( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Normal image: %s", g_input_normal.filename.c_str());
            g_input_normal.data = new OIIO::ImageBuf(g_input_normal.filename);
            if (g_input_normal.data->init_spec(g_input_normal.filename, 0, 0))
            {
                if (g_verbosity >= 2)
                    PrintInfo("Loaded successfully");
                n_loaded = true;
            }
            else
            {
                PrintError("Failed to load normal image");
                PrintError("[OIIO]: %s", g_input_normal.data->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if (arg == "-a")
        {
            i++;
            g_input_albedo.filename = std::string( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Albedo image: %s", g_input_albedo.filename.c_str());
            g_input_albedo.data = new OIIO::ImageBuf(g_input_albedo.filename);
            if (g_input_albedo.data->init_spec(g_input_albedo.filename, 0, 0))
            {
                if (g_verbosity >= 2)
                    PrintInfo("Loaded successfully");
                a_loaded = true;
            }
            else
            {
                PrintError("Failed to load albedo image");
                PrintError("[OIIO]: %s", g_input_albedo.data->geterror().c_str());
                cleanup();
                exitfunc(EXIT_FAILURE);
            }
        }
        else if(arg == "-o")
        {
            i++;
            g_input_beauty.output_filename = std::string( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Output image: %s", g_input_beauty.output_filename.c_str());
        }
        else if(arg == "-os")
        {
            i++;
            out_suffix = std::string( argv[i] );
            if (g_verbosity >= 2)
                PrintInfo("Output suffix: %s", out_suffix.c_str());
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
    if (!g_input_beauty.output_filename.empty())
    {
        int ext_loc = (int)g_input_beauty.output_filename.find_last_of(".");
        ext_loc++;
        const char* ext_c = g_input_beauty.output_filename.c_str()+ext_loc;
        std::string ext(ext_c);
        if (!ext.size())
        {
            PrintError("No output file extension");
            cleanup();
            exitfunc(EXIT_FAILURE);
        }
    }

    // If no output path then we need an suffix for the new file
    if (g_input_beauty.output_filename.empty() && out_suffix.empty())
    {
        PrintError("No output suffix set");
        cleanup();
        exitfunc(EXIT_FAILURE);
    }


    OIIO::ROI beauty_roi, albedo_roi, normal_roi;
    beauty_roi = OIIO::get_roi_full(g_input_beauty.data->spec());
    int b_width = beauty_roi.width();
    int b_height = beauty_roi.height();
    if (a_loaded)
    {
        albedo_roi = OIIO::get_roi_full(g_input_albedo.data->spec());
        if (n_loaded)
            normal_roi = OIIO::get_roi_full(g_input_normal.data->spec());
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

    for (auto& it : g_input_aov)
    {
        auto& a = it.second;
        OIIO::ROI aov_roi = OIIO::get_roi_full(a.data->spec());
        if (aov_roi.width() != b_width || aov_roi.height() != b_height)
        {
            PrintError("AOV image %s not same resolution as beauty", a.filename.c_str());
            cleanup();
            exitfunc(EXIT_FAILURE);
        }
    }

    // Allocate space for our pixel data
    std::vector<float> beauty_pixels(b_width * b_height * beauty_roi.nchannels());
    std::vector<std::vector<float> > aov_pixels(g_input_aov.size());

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
    result = optixDeviceContextCreate(cuCtx, nullptr, &optix_context);
    if (result != OPTIX_SUCCESS)
    {
        PrintError("Could not create OptiX context: (%d) %s", result, optixGetErrorName(result));
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
    if (denoise_aovs)
        model = OPTIX_DENOISER_MODEL_KIND_AOV;
    OPTIX_CHECK( optixDenoiserCreate(optix_context, model, &denoiser_options, &optix_denoiser) );
    // Compute memory needed for the denoiser to exist on the GPU
    OptixDenoiserSizes denoiser_sizes;
    memset(&denoiser_sizes, 0, sizeof(OptixDenoiserSizes));
    OPTIX_CHECK( optixDenoiserComputeMemoryResources(optix_denoiser, b_width, b_height, &denoiser_sizes) );
    // Allocate this space on the GPu
    void* denoiser_state_buffer = nullptr;
    void* denoiser_scratch_buffer = nullptr;
    CU_CHECK(cudaMalloc(&denoiser_state_buffer, denoiser_sizes.stateSizeInBytes));
    CU_CHECK(cudaMalloc(&denoiser_scratch_buffer, denoiser_sizes.withoutOverlapScratchSizeInBytes));
    // Setup the denoiser
    OPTIX_CHECK( optixDenoiserSetup(optix_denoiser, cuda_stream,
                                            b_width, b_height,
                                            (CUdeviceptr)denoiser_state_buffer,   denoiser_sizes.stateSizeInBytes,
                                            (CUdeviceptr)denoiser_scratch_buffer, denoiser_sizes.withoutOverlapScratchSizeInBytes) );

    // Set the denoiser parameters
    OptixDenoiserParams denoiser_params = {};
    // TODO: Expose option for this
    denoiser_params.denoiseAlpha = 0;
    denoiser_params.blendFactor = blend;
    CU_CHECK(cudaMalloc((void**)&denoiser_params.hdrIntensity, sizeof(float)));

    // Create and set our OptiX layers
    std::vector<OptixDenoiserLayer> layers(1 + g_input_aov.size());
    memset (&layers[0], 0, sizeof(OptixDenoiserLayer) * layers.size());

    // Allocate memory for all our layers on the GPU
    for (auto& l : layers)
    {
        // Input
        CU_CHECK(cudaMalloc(&((void*)l.input.data), sizeof(float) * 4 * b_width * b_height));
        l.input.width              = b_width;
        l.input.height             = b_height;
        l.input.rowStrideInBytes   = b_width * sizeof(float) * 4;
        l.input.pixelStrideInBytes = sizeof(float) * 4;
        l.input.format             = OPTIX_PIXEL_FORMAT_FLOAT4;

        // Output
        CU_CHECK(cudaMalloc(&((void*)l.output.data), sizeof(float) * 4 * b_width * b_height));
        l.output.width              = b_width;
        l.output.height             = b_height;
        l.output.rowStrideInBytes   = b_width * sizeof(float) * 4;
        l.output.pixelStrideInBytes = sizeof(float) * 4;
        l.output.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
    }

    OptixDenoiserGuideLayer guide_layer = {};
    // albedo
    if (a_loaded)
    {
        CU_CHECK(cudaMalloc(&((void*)guide_layer.albedo.data), sizeof(float) * 4 * b_width * b_height));
        // guide_layer.albedo.data               = (CUdeviceptr)albedo_buffer;
        guide_layer.albedo.width              = a_width;
        guide_layer.albedo.height             = a_height;
        guide_layer.albedo.rowStrideInBytes   = a_width * sizeof(float) * 4;
        guide_layer.albedo.pixelStrideInBytes = sizeof(float) * 4;
        guide_layer.albedo.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
    }

    // normal
    if (n_loaded)
    {
        CU_CHECK(cudaMalloc(&((void*)guide_layer.normal.data), sizeof(float) * 4 * b_width * b_height));
        // guide_layer.normal.data               = (CUdeviceptr)normal_buffer;
        guide_layer.normal.width              = n_width;
        guide_layer.normal.height             = n_height;
        guide_layer.normal.rowStrideInBytes   = n_width * sizeof(float) * 4;
        guide_layer.normal.pixelStrideInBytes = sizeof(float) * 4;
        guide_layer.normal.format             = OPTIX_PIXEL_FORMAT_FLOAT4;
    }

    unsigned int buffer_size = 4 * b_width * b_height;
    std::vector<float> host_scratch(buffer_size, 0.f);

    // Copy our beauty image data to the GPU
    // Convert image to float4 to use with the denoiser
    g_input_beauty.data->get_pixels(beauty_roi, OIIO::TypeDesc::FLOAT, &beauty_pixels[0]);
    imageConvertFormat(&beauty_pixels[0], beauty_roi.nchannels(), &host_scratch[0], 4, b_width, b_height);
    // Copy our data to the GPU
    // First layer must always be beauty AOV
    CU_CHECK(cudaMemcpy((void*)layers[0].input.data, &host_scratch[0], sizeof(float) * buffer_size, cudaMemcpyHostToDevice));

    if (a_loaded)
    {
        // Copy albedo image data to the GPU
        std::vector<float> albedo_pixels(a_width * a_height * albedo_roi.nchannels());
        g_input_albedo.data->get_pixels(albedo_roi, OIIO::TypeDesc::FLOAT, &albedo_pixels[0]);
        memset(&host_scratch[0], 0, sizeof(float) * buffer_size);
        // Convert image to float4 to use with the denoiser
        imageConvertFormat(&albedo_pixels[0], albedo_roi.nchannels(), &host_scratch[0], 4, a_width, a_height);
        // Copy our data to the GPU
        CU_CHECK(cudaMemcpy((void*)guide_layer.albedo.data, &host_scratch[0], sizeof(float) * buffer_size, cudaMemcpyHostToDevice));
    }

    if (n_loaded)
    {
        // Copy normal image data to the GPU
        std::vector<float> normal_pixels(n_width * n_height * normal_roi.nchannels());
        g_input_normal.data->get_pixels(normal_roi, OIIO::TypeDesc::FLOAT, &normal_pixels[0]);
        memset(&host_scratch[0], 0, sizeof(float) * buffer_size);
        // Convert image to float4 to use with the denoiser
        imageConvertFormat(&normal_pixels[0], normal_roi.nchannels(), &host_scratch[0], 4, n_width, n_height);
        // Copy our data to the GPU
        CU_CHECK(cudaMemcpy((void*)guide_layer.normal.data, &host_scratch[0], sizeof(float) * buffer_size, cudaMemcpyHostToDevice));
    }

    // Any additional AOVs that need to be denoised
    int aov = 1;
    for (auto& it : g_input_aov)
    {
        auto& a = it.second;
        // Copy normal image data to the GPU
        OIIO::ROI aov_roi = OIIO::get_roi_full(a.data->spec());
        aov_pixels[aov - 1] = std::vector<float>(albedo_roi.width() * albedo_roi.height() * aov_roi.nchannels());
        a.data->get_pixels(aov_roi, OIIO::TypeDesc::FLOAT, &aov_pixels[aov - 1][0]);
        memset(&host_scratch[0], 0, sizeof(float) * buffer_size);
        // Convert image to float4 to use with the denoiser
        imageConvertFormat(&aov_pixels[aov - 1][0], aov_roi.nchannels(), &host_scratch[0], 4, albedo_roi.width(), albedo_roi.height());
        // Copy our data to the GPU
        CU_CHECK(cudaMemcpy((void*)layers[aov].input.data, &host_scratch[0], sizeof(float) * buffer_size, cudaMemcpyHostToDevice));
        aov++;
    }

    // Execute denoiser
    int sum = 0;
    for (unsigned int i = 0; i < num_runs; i++)
    {
        PrintInfo("Denoising...");
        clock_t start = clock(), diff;
        // Compute the intensity of the input image
        OPTIX_CHECK( optixDenoiserComputeIntensity(optix_denoiser, cuda_stream, &layers[0].input, denoiser_params.hdrIntensity,
                                                        (CUdeviceptr)denoiser_scratch_buffer, denoiser_sizes.withoutOverlapScratchSizeInBytes) );

        // Execute the denoiser
        OPTIX_CHECK( optixDenoiserInvoke(optix_denoiser, cuda_stream, &denoiser_params,
                                            (CUdeviceptr)denoiser_state_buffer, denoiser_sizes.stateSizeInBytes,
                                            &guide_layer, &layers[0], layers.size(), 0, 0,
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

    // Copy denoised images back to the CPU
    unsigned int num_layers = layers.size();
    auto aov_it = g_input_aov.begin();
    for (unsigned int i = 0; i < num_layers; i++)
    {
        std::vector<float>& output = (i == 0) ? beauty_pixels : aov_pixels[0];
        unsigned int num_channels = 0;
        if (i == 0)
        {
            num_channels = beauty_roi.nchannels();
        }
        else
        {
            OIIO::ROI aov_roi = OIIO::get_roi_full(aov_it->second.data->spec());
            num_channels = aov_roi.nchannels();
            aov_it++;
        }
        CU_CHECK(cudaMemcpy(&host_scratch[0], (void*)layers[i].output.data, sizeof(float) * buffer_size, cudaMemcpyDeviceToHost));
        imageConvertFormat(&host_scratch[0], 4, &output[0], num_channels, b_width, b_height);
    }


    // Remove our gpu buffers
    for (auto& l : layers)
    {
        CU_CHECK(cudaFree((void*)l.input.data));
        CU_CHECK(cudaFree((void*)l.output.data));
    }
    CU_CHECK(cudaFree((void*)denoiser_params.hdrIntensity));
    CU_CHECK(cudaFree((void*)guide_layer.albedo.data));
    CU_CHECK(cudaFree((void*)guide_layer.normal.data));
    // Destroy the denoiser
    CU_CHECK(cudaFree(denoiser_state_buffer));
    CU_CHECK(cudaFree(denoiser_scratch_buffer));
    OPTIX_CHECK( optixDenoiserDestroy(optix_denoiser) );
    // Destroy the OptiX context
    OPTIX_CHECK( optixDeviceContextDestroy(optix_context) );
    // Delete our CUDA stream as well
    CU_CHECK(cudaStreamDestroy(cuda_stream));


    // If the image already exists delete it
    remove(g_input_beauty.output_filename.c_str());

    // Set our OIIO pixels
    if (!g_input_beauty.data->set_pixels(beauty_roi, OIIO::TypeDesc::FLOAT, &beauty_pixels[0]))
        PrintError("Something went wrong setting pixels of file %s", g_input_beauty.filename.c_str());

    // Save the output image
    std::string out_path = g_input_beauty.output_filename;
    if (!out_suffix.empty())
    {
        if (g_input_beauty.output_filename.empty())
            g_input_beauty.output_filename = g_input_beauty.filename;
        int ext_loc = (int)g_input_beauty.output_filename.find_last_of(".");
        const char* ext_c = g_input_beauty.output_filename.c_str()+ext_loc;
        std::string ext(ext_c);
        out_path = g_input_beauty.output_filename.substr(0, ext_loc) + out_suffix + ext_c;
    }
    if (g_input_beauty.data->write(out_path))
        PrintInfo("Written out: %s", out_path.c_str());
    else
    {
        PrintError("Could not save file %s", out_path.c_str());
        PrintError("[OIIO]: %s", g_input_beauty.data->geterror().c_str());
    }

    // Save all the additional AOVs that have been denoised
    aov = 0;
    for (auto& it : g_input_aov)
    {
        auto& a = it.second;
        // Set our OIIO pixels
        OIIO::ROI aov_roi = OIIO::get_roi_full(a.data->spec());
        if (!a.data->set_pixels(aov_roi, OIIO::TypeDesc::FLOAT, &aov_pixels[aov][0]))
            PrintError("Something went wrong setting pixels of file %s", a.filename.c_str());
        std::string out_path = a.output_filename;
        int ext_loc = (int)a.output_filename.find_last_of(".");
        const char* ext_c = out_path.c_str()+ext_loc;
        std::string ext(ext_c);
        out_path = out_path.substr(0, ext_loc) + out_suffix + ext_c;

        if (a.data->write(out_path))
            PrintInfo("Written out: %s", out_path.c_str());
        else
        {
            PrintError("Could not save file %s", out_path.c_str());
            PrintError("[OIIO]: %s", a.data->geterror().c_str());
        }
        aov++;
    }

    PrintInfo("Done!");
    cleanup();
    exitfunc(EXIT_SUCCESS);
}

