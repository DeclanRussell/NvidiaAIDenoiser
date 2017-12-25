#include <QImage>
#include <optix_world.h>
#include <iostream>

int main(int argc, char *argv[])
{
    std::cout<<"Launching Nvidia AI Denoiser command line app v1.0"<<std::endl;
    std::cout<<"Created by Declan Russell (25/12/2017 ~ Merry Christmas!)"<<std::endl;

    QImage input_beauty;
    QImage input_normal;
    QImage input_albedo;
    std::string out_path;
    float blend = 0.f;
    for (int i=1; i<argc; i++)
    {
        const std::string arg( argv[i] );
        if (arg == "-i")
        {
            i++;
            std::string path( argv[i] );
            input_beauty = QImage(path.c_str());
            std::cout<<"Input image: "<<path<<std::endl;
        }
        else if (arg == "-n")
        {
            i++;
            std::string path( argv[i] );
            input_normal = QImage(path.c_str());
            std::cout<<"Normal image: "<<path<<std::endl;
        }
        else if (arg == "-a")
        {
            i++;
            std::string path( argv[i] );
            input_albedo = QImage(path.c_str());
            std::cout<<"Albedo image: "<<path<<std::endl;
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
            std::cout<<"-a [string] : path to input albedo AOV"<<std::endl;
            std::cout<<"-n [string] : path to input normal AOV"<<std::endl;
            std::cout<<"-b [float] : blend amount"<<std::endl;
        }
    }

    if (input_beauty.isNull())
    {
        std::cerr<<"No input image could be loaded";
        return EXIT_FAILURE;
    }

    if (!input_normal.isNull() && input_albedo.isNull())
    {
        std::cerr<<"You cannot use a normal AOV without an albedo";
        return EXIT_FAILURE;
    }

    // Get the file extension
    int x = out_path.find_last_of(".");
    x++;
    const char* ext_c = out_path.c_str()+x;
    std::string ext(ext_c);
    if (!ext.size())
    {
        std::cerr<<"No output file extension";
        return EXIT_FAILURE;
    }

    int width, height, normal_width, normal_height;
    width = input_beauty.width();
    height = input_beauty.height();
    normal_width = normal_height = 0;
    if (!input_albedo.isNull())
    {
        normal_width = input_normal.width();
        normal_height = input_normal.height();
    }

    QImage out_image(width, height, input_beauty.format());

    optix::Context optix_context = optix::Context::create();
    optix::Buffer beauty_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);
    optix::Buffer albedo_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, input_albedo.width(), input_albedo.height());
    optix::Buffer normal_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, normal_width, normal_height);
    optix::Buffer out_buffer = optix_context->createBuffer(RT_BUFFER_INPUT_OUTPUT, RT_FORMAT_FLOAT4, width, height);

    optix::float4* ptr = (optix::float4*)beauty_buffer->map();
    for(int y=0; y<height; y++)
    for(int x=0; x<width; x++)
    {
        QColor c = input_beauty.pixelColor(x,y);
        ptr[y*width+x] = optix::make_float4(c.redF(), c.greenF(), c.blueF(), c.alphaF());
    }
    beauty_buffer->unmap();

    if (!input_albedo.isNull())
    {
        ptr = (optix::float4*)albedo_buffer->map();
        for(int y=0; y<height; y++)
        for(int x=0; x<width; x++)
        {
            QColor c = input_albedo.pixelColor(x,y);
            ptr[y*width+x] = optix::make_float4(c.redF(), c.greenF(), c.blueF(), c.alphaF());
        }
        albedo_buffer->unmap();
    }

    if (!input_albedo.isNull() && !input_normal.isNull())
    {
        ptr = (optix::float4*)normal_buffer->map();
        for(int y=0; y<height; y++)
        for(int x=0; x<width; x++)
        {
            QColor c = input_normal.pixelColor(x,y);
            ptr[y*width+x] = optix::make_float4(c.redF(), c.greenF(), c.blueF(), c.alphaF());
        }
        normal_buffer->unmap();
    }

    optix::PostprocessingStage denoiserStage = optix_context->createBuiltinPostProcessingStage("DLDenoiser");
    denoiserStage->declareVariable("input_buffer")->set(beauty_buffer);
    denoiserStage->declareVariable("output_buffer")->set(out_buffer);
    denoiserStage->declareVariable("blend")->setFloat(blend);
    denoiserStage->declareVariable("input_albedo_buffer")->set(albedo_buffer);
    denoiserStage->declareVariable("input_normal_buffer")->set(normal_buffer);

    optix::CommandList commandList= optix_context->createCommandList();
    commandList->appendPostprocessingStage(denoiserStage, width, height);
    commandList->finalize();

    optix_context->validate();
    optix_context->compile();

    std::cout<<"Denoising..."<<std::endl;
    commandList->execute();
    std::cout<<"Denoising complete"<<std::endl;

    ptr = (optix::float4*)out_buffer->map();
    for(int y=0; y<height; y++)
    for(int x=0; x<width; x++)
    {
        optix::float4 v = ptr[y*width+x];
        // TODO: QImage requires clamped values between 0-1. Ideally we need a better
        //       image library what can deal with these cases.
        v = optix::clamp(v, optix::make_float4(0.f,0.f,0.f,0.f), optix::make_float4(1.f,1.f,1.f,1.f));
        QColor c;
        c.setRedF(v.x);
        c.setGreenF(v.y);
        c.setBlueF(v.z);
        c.setAlphaF(v.w);
        out_image.setPixelColor(x, y, c);
    }
    out_buffer->unmap();

    beauty_buffer->destroy();
    normal_buffer->destroy();
    albedo_buffer->destroy();
    optix_context->destroy();

    std::cout<<"Saving to: "<<out_path<<std::endl;
    if (out_image.save(out_path.c_str(), ext.c_str()))
        std::cout<<"Done!"<<std::endl;
    else
        std::cerr<<"Could not save file"<<std::endl;

    return EXIT_SUCCESS;
}

