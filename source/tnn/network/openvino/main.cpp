#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

#include <memory>
#include <string>
#include <chrono>

/* create function
          |    input     |
      /      |      |      \
    split0 split1 split2 split3
       \     /      |       |
         add        |       |
          |         |       |
         res0     res1     res2
*/
std::shared_ptr<ngraph::Function> create_function() {
    auto input_node = std::make_shared<ngraph::op::Parameter>(
        ngraph::element::Type_t::f32, ngraph::Shape(std::vector<size_t>{{8,1,4,4}}));
    input_node->set_friendly_name("input");

    auto input_size = input_node->get_output_shape(0).size();
    
    std::vector<int> begins, ends;
    std::vector<int64_t> begin_mask, end_mask;
    ngraph::Shape dims_shape({input_size});
    std::vector<int> slices(4, 2);

    ngraph::NodeVector split_nodes;

    for (int i = 0; i < input_size; i++) {
        if (i == 0) {
            begin_mask.push_back(0);
            end_mask.push_back(0);
        } else {
            begin_mask.push_back(1);
            end_mask.push_back(1);
        }
        begins.push_back(0);
        ends.push_back(0);
    }

    for (int i = 0; i < slices.size(); i++) {
        begins[0] = ends[0];
        ends[0]  += slices[i];

        auto beginNode = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::i32, dims_shape, begins);
        auto endNode   = std::make_shared<ngraph::op::Constant>(
            ngraph::element::Type_t::i32, dims_shape, ends);
        
        auto split_node = std::make_shared<ngraph::op::v1::StridedSlice>(
            input_node->output(0), beginNode, endNode, begin_mask, end_mask);
        
        split_node->validate_and_infer_types();
        split_node->set_friendly_name("split" + std::to_string(i));

        split_nodes.emplace_back(split_node);
    }

    auto add_node = std::make_shared<ngraph::op::v1::Add>(split_nodes[0]->output(0), split_nodes[1]->output(0));
    add_node->validate_and_infer_types();
    add_node->set_friendly_name("add");

    auto res0 = std::make_shared<ngraph::op::Result>(add_node->output(0));
    auto res1 = std::make_shared<ngraph::op::Result>(split_nodes[2]->output(0));
    auto res2 = std::make_shared<ngraph::op::Result>(split_nodes[3]->output(0));
    ngraph::NodeVector output_nodes = {res0, res1, res2};

    std::shared_ptr<ngraph::Function> fn_ptr = std::make_shared<ngraph::Function>(output_nodes, ngraph::ParameterVector{input_node}, "Debug");
    return fn_ptr;
}

int main(int argc, char *argv[]) {
    InferenceEngine::CNNNetwork network(create_function());
    InferenceEngine::Core ie;

    std::map<std::string, std::string> config = {
        {CONFIG_KEY(CPU_THREADS_NUM), "1"},
        {CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "0"},
        {CONFIG_KEY(CPU_BIND_THREAD), "NO"},
    };

    ie.SetConfig(config, "CPU");

    if (argc == 2) {
        // read onnx model
        network = ie.ReadNetwork(argv[1]);
    }
    if (argc == 3) {
        // read xml/bin
        network = ie.ReadNetwork(argv[1], argv[2]);
    }

    InferenceEngine::ExecutableNetwork exec_network = ie.LoadNetwork(network, "CPU");

    InferenceEngine::InferRequest infer_request = exec_network.CreateInferRequest();

    InferenceEngine::InputsDataMap input_info = network.getInputsInfo();
    InferenceEngine::OutputsDataMap output_info = network.getOutputsInfo();

    // for (auto &input : input_info) {
    //     input.second->setPrecision(InferenceEngine::Precision::FP32);
    //     input.second->setLayout(InferenceEngine::Layout::NCHW);
    // }

    // for (auto &output : input_info) {
    //     output.second->setPrecision(InferenceEngine::Precision::FP32);
    //     output.second->setLayout(InferenceEngine::Layout::NCHW);
    // }

    // ie = InferenceEngine::Core();
    // config = {
    //     {CONFIG_KEY(CPU_THREADS_NUM), "4"},
    //     {CONFIG_KEY(CPU_THROUGHPUT_STREAMS), "0"},
    //     {CONFIG_KEY(CPU_BIND_THREAD), "NO"},
    // };

    // ie.SetConfig(config, "CPU");
    // exec_network = ie.LoadNetwork(network, "CPU");
    // infer_request = exec_network.CreateInferRequest();

    const int LOOP = 1;
    auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < LOOP; i++)
        infer_request.Infer();
    auto t2 = std::chrono::steady_clock::now();
    std::cout << "Infer Time:     " << (std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count())/1000.0/LOOP
                  << " ms.\n\n";
    return 0;
}
