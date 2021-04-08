#include <fstream>
#include <string>
#include <vector>

#include "macro.h"
#include "utils/utils.h"

#include "test_classifier.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace TNN_NS;

int main(int argc, char** argv) {

    if (argc < 3) {
        printf("how to run:  %s proto model height width\n", argv[0]);
        return -1;
    }
    // 创建tnn实例
    auto proto_content = fdLoadFile(argv[1]);
    auto model_content = fdLoadFile(argv[2]);
    
    int h = 288, w = 160;
    // if(argc >= 5) {
    //     h = std::atoi(argv[3]);
    //     w = std::atoi(argv[4]);
    // }
    auto option = std::make_shared<TNNSDKOption>();
    {
        option->proto_content = proto_content;
        option->model_content = model_content;
        option->library_path = "";

        option->compute_units = TNN_NS::TNNComputeUnitsCPU;
    }

    auto predictor = std::make_shared<TestClassifier>();
    std::vector<int> nchw = {1, 3, h, w};

    char* temp_p;
    char line[256];
    
    char img_buff[256];
    char *input_imgfn = img_buff;
    // if(argc < 6)
        // strncpy(input_imgfn, "../../assets/leijun.png", 256);
    // else
    
    strncpy(input_imgfn, argv[3], 256);

    printf("Classify is about to start, and the picrture is %s\n",input_imgfn);

    int image_width, image_height, image_channel;
    unsigned char *data = stbi_load(input_imgfn, &image_width, &image_height, &image_channel, 3);

    //Init
    std::shared_ptr<TNNSDKOutput> sdk_output = predictor->CreateSDKOutput();
    CHECK_TNN_STATUS(predictor->Init(option));
    //Predict
    auto image_mat = std::make_shared<TNN_NS::Mat>(TNN_NS::DEVICE_ARM, TNN_NS::N8UC3, nchw, data);
    
    for (int i = 0;i<1000;i++) {
    	CHECK_TNN_STATUS(predictor->Predict(std::make_shared<TNNSDKInput>(image_mat), sdk_output));
    }

    int class_id = -1;
    if (sdk_output && dynamic_cast<TestClassifierOutput *>(sdk_output.get())) {
        auto classfy_output = dynamic_cast<TestClassifierOutput *>(sdk_output.get());
        class_id = classfy_output->class_id;
    }
    //完成计算，获取任意输出点
    fprintf(stdout, "Classify done. \n");
    free(data);
    return 0;

}
