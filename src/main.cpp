
#include <optix_world.h>
#include <iostream>
#include <IL/il.h>
#include <IL/ilu.h>
#include <stdio.h>
#include <exception>

// Our global image handles
ILuint input_beauty;
ILuint input_normal;
ILuint input_albedo;
ILuint out_image;

void cleanup()
{
    ilDeleteImages(1, &input_beauty);
    ilDeleteImages(1, &input_normal);
    ilDeleteImages(1, &input_albedo);
    ilDeleteImages(1, &out_image);
    ilShutDown();
}

int main(int argc, char *argv[])
{
    std::cout<<"Launching Nvidia AI Denoiser command line app v1.1"<<std::endl;
    std::cout<<"Created by Declan Russell (25/12/2017 ~ Merry Christmas!)"<<std::endl;

    ilInit();
    iluInit();

    ilGenImages(1, &input_beauty);
    ilGenImages(1, &input_normal);
    ilGenImages(1, &input_albedo);
    ilGenImages(1, &out_image);

    ILboolean b_loaded, n_loaded, a_loaded;
    b_loaded = n_loaded = a_loaded = false;

    // Pass our command line args
    std::string out_path;
    float blend = 0.f;
    for (int i=1; i<argc; i++)
    {
        const std::string arg( argv[i] );
        if (arg == "-i")
        {
            i++;
            std::string path( argv[i] );
            ilBindImage(input_beauty);
            b_loaded = ilLoadImage(path.c_str());
            std::cout<<"Input image: "<<path<<std::endl;
            if (b_loaded)
                std::cout<<"Loaded successfully"<<std::endl;
            else
            {
                std::cout<<"Failed to load input image"<<std::endl;
                ILenum error = ilGetError();
                const char* str;
                while (error != IL_NO_ERROR)
                {
                    str = iluErrorString(error);
                    std::cout<<"[DevIL]: "<<str<<std::endl;
                    error = ilGetError();
                }
            }
            iluFlipImage();
        }
        else if (arg == "-n")
        {
            i++;
            std::string path( argv[i] );
            ilBindImage(input_normal);
            n_loaded = ilLoadImage(path.c_str());
            std::cout<<"Normal image: "<<path<<std::endl;
            if (n_loaded)
                std::cout<<"Loaded successfully"<<std::endl;
            else
            {
                std::cout<<"Failed to load normal image"<<std::endl;
                ILenum error = ilGetError();
                const char* str;
                while (error != IL_NO_ERROR)
                {
                    str = iluErrorString(error);
                    std::cout<<"[DevIL]: "<<str<<std::endl;
                    error = ilGetError();
                }
            }
            iluFlipImage();
        }
        else if (arg == "-a")
        {
            i++;
            std::string path( argv[i] );
            ilBindImage(input_albedo);
            a_loaded = ilLoadImage(path.c_str());
            std::cout<<"Albedo image: "<<path<<std::endl;
            if (a_loaded)
                std::cout<<"Loaded successfully"<<std::endl;
            else
            {
                std::cout<<"Failed to load albedo image"<<std::endl;
                ILenum error = ilGetError();
                const char* str;
                while (error != IL_NO_ERROR)
                {
                    str = iluErrorString(error);
                    std::cout<<"[DevIL]: "<<str<<std::endl;
                    error = ilGetError();
                }
            }
            iluFlipImage();
        }
        else if(arg == "-o")
        {
            i++;
            out_path = std::string( argv[i] );
            std::cout<<"Output image: "<<out_path<<std::endl;
        }
        else if (arg == "-b")
        {
            i++;
            std::string blend_string( argv[i] );
            blend = std::stof(blend_string);
            std::cout<<"Blend amount: "<<blend<<std::endl;
        }
        else if (arg == "-h" || arg == "--help")
        {
            std::cout<<"Command line parameters"<<std::endl;
            std::cout<<"-i [string] : path to input image"<<std::endl;
            std::cout<<"-o [string] : path to output image"<<std::endl;
            std::cout<<"-a [string] : path to input albedo AOV"<<std::endl;
            std::cout<<"-n [string] : path to input normal AOV"<<std::endl;
            std::cout<<"-b [float] : blend amount"<<std::endl;
        }
    }

    if (!b_loaded)
    {
        std::cerr<<"No input image could be loaded"<<std::endl;
        cleanup();
        return EXIT_FAILURE;
    }

    if (n_loaded && !a_loaded)
    {
        std::cerr<<"You cannot use a normal AOV without an albedo"<<std::endl;
        cleanup();
        return EXIT_FAILURE;
    }

    // Check for a file extension
    int x = (int)out_path.find_last_of(".");
    x++;
    const char* ext_c = out_path.c_str()+x;
    std::string ext(ext_c);
    if (!ext.size())
    {
        std::cerr<<"No output file extension"<<std::endl;
        cleanup();
        return EXIT_FAILURE;
    }

    // Prepare our image data
    ilBindImage(input_beauty);
    int b_width = ilGetInteger(IL_IMAGE_WIDTH);
    int b_height = ilGetInteger(IL_IMAGE_HEIGHT);
    ilConvertImage(IL_RGBA, IL_FLOAT);
    int bpp = ilGetInteger((IL_IMAGE_BPP));

    int a_width = 0;
    int a_height = 0;
    if (a_loaded)
    {
        ilBindImage(input_albedo);
        a_width = ilGetInteger(IL_IMAGE_WIDTH);
        a_height = ilGetInteger(IL_IMAGE_HEIGHT);
        if (a_width != b_width || a_height != b_height)
        {
            std::cerr<<"Aldedo image not same resolution as beauty"<<std::endl;
            cleanup();
            return EXIT_FAILURE;
        }
        ilConvertImage(IL_RGBA, IL_FLOAT);
    }

    int n_width = 0;
    int n_height = 0;
    if (n_loaded)
    {
        ilBindImage(input_normal);
        n_width = ilGetInteger(IL_IMAGE_WIDTH);
        n_height = ilGetInteger(IL_IMAGE_HEIGHT);
        if (n_width != b_width || n_height != b_height)
        {
            std::cerr<<"Normal image not same resolution as beauty"<<std::endl;
            cleanup();
            return EXIT_FAILURE;
        }
        ilConvertImage(IL_RGBA, IL_FLOAT);
    }

    // Catch optix exceptions
    try
    {
        // Create our optix context and image buffers
        optix::Context optix_context = optix::Context::create();
        optix::Buffer beauty_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, b_width, b_height);
        optix::Buffer albedo_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, a_width, a_height);
        optix::Buffer normal_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, n_width, n_height);
        optix::Buffer out_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, b_width, b_height);

        // Copy all our image data to the gpu buffers
        ilBindImage(input_beauty);
        ILubyte* data = ilGetData();
        memcpy(beauty_buffer->map(), data, sizeof(float) * 4 * b_width * b_height);
        beauty_buffer->unmap();

        if (a_loaded)
        {
            ilBindImage(input_albedo);
            ILubyte* data = ilGetData();
            memcpy(albedo_buffer->map(), data, sizeof(float) * 4 * b_width * b_height);
            albedo_buffer->unmap();
        }

        if (a_loaded && n_loaded)
        {
            ilBindImage(input_normal);
            ILubyte* data = ilGetData();
            memcpy(normal_buffer->map(), data, sizeof(float) * 4 * b_width * b_height);
            normal_buffer->unmap();
        }

        // Setup the optix denoiser post processing stage
        optix::PostprocessingStage denoiserStage = optix_context->createBuiltinPostProcessingStage("DLDenoiser");
        denoiserStage->declareVariable("input_buffer")->set(beauty_buffer);
        denoiserStage->declareVariable("output_buffer")->set(out_buffer);
        denoiserStage->declareVariable("blend")->setFloat(blend);
        denoiserStage->declareVariable("input_albedo_buffer")->set(albedo_buffer);
        denoiserStage->declareVariable("input_normal_buffer")->set(normal_buffer);

        // Add the denoiser to the new optix command list
        optix::CommandList commandList= optix_context->createCommandList();
        commandList->appendPostprocessingStage(denoiserStage, b_width, b_height);
        commandList->finalize();

        // Compile our context. I'm not sure if this is needed given there is no megakernal?
        optix_context->validate();
        optix_context->compile();

        // Execute denoise
        std::cout<<"Denoising..."<<std::endl;
        commandList->execute();
        std::cout<<"Denoising complete"<<std::endl;

        // Create our ouput image
        ilBindImage(out_image);
        ilTexImage(b_width, b_height, 0, bpp, IL_RGBA, IL_FLOAT, NULL);

        // Copy denoised image back to the cpu
        ilBindImage(out_image);
        data = ilGetData();
        memcpy(data, out_buffer->map(), sizeof(float) * 4 * b_width * b_height);
        out_buffer->unmap();

        // Remove our gpu buffers
        beauty_buffer->destroy();
        normal_buffer->destroy();
        albedo_buffer->destroy();
        optix_context->destroy();
                                            
    }                                               
    catch (std::exception e)                        
    {                                               
        std::cerr<<"[OptiX]: "<<e.what()<<std::endl;
        cleanup();                                  
        return EXIT_FAILURE;                        
    }


    // If the image already exists delete it
    remove(out_path.c_str());

    // Save the output image
    std::cout<<"Saving to: "<<out_path<<std::endl;
    if (ilSaveImage(out_path.c_str()))
        std::cout<<"Done!"<<std::endl;
    else
        std::cerr<<"Could not save file "<<out_path<<" ("<<iluErrorString(ilGetError())<<")"<<std::endl;

    cleanup();
    return EXIT_SUCCESS;
}

