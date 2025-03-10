#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>

#include <marisa.h>

using namespace std;

namespace py = pybind11;

class FastTrie {
public:
    FastTrie(const vector<string>& keys) {
        marisa::Keyset keyset;
        for (const string& k : keys) {
            const char* char_arr = k.c_str();
            keyset.push_back(char_arr);
        }
        trie.build(keyset);
    }
    
    vector<int> get_valid_tokens(const string& remaining_text) {
        const char* input_arr = remaining_text.c_str();

        marisa::Agent agent;
        agent.set_query(input_arr);
        vector<int> result;
        while (trie.common_prefix_search(agent)) {
            result.push_back(agent.key().id());
        }
        
        return result;
    }
    
private:
    marisa::Trie trie;
};

// The macro below defines a Python module named "marisa_ext".
// You can change "marisa_ext" to whatever module name you want.
PYBIND11_MODULE(marisa_ext, m) {
    m.doc() = "Python bindings for MARISA trie functions";
    // m.def("get_num_results", &get_num_results,
    //       "Return the number of common prefix matches in the MARISA trie");
    // m.def("get_valid_tokens", &get_valid_tokens, "whatever");
    py::class_<FastTrie>(m, "FastTrie")
        .def(py::init<const std::vector<std::string>&>(), py::arg("keys"),
             "Construct a FastTrie from a list of keys")
        .def("get_valid_tokens", &FastTrie::get_valid_tokens, py::arg("remaining_text"),
             "Return a list of valid token ids for the given remaining text");
}
