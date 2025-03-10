#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <map>

#include <xcdat.hpp>

using namespace std;

namespace py = pybind11;

class FastTrie {
public:
    FastTrie(const vector<string>& keys) {
        int n = keys.size();
        map<string, int> key_to_token_id;
        for (int i = 0; i < n; ++i) {
            key_to_token_id[keys[i]] = i;
        }
        n = key_to_token_id.size();
        vector<string> sorted_keys;
        sorted_keys.reserve(n);
        for (const auto& p : key_to_token_id) {
            sorted_keys.push_back(p.first);
        }
        trie = xcdat::trie_8_type(sorted_keys);
        trie_id_to_token_id = vector<int>(n);
        for (const auto& p : key_to_token_id) {
            uint64_t trie_id = trie.lookup(p.first).value_or(UINT64_MAX);
            assert(trie_id != UINT64_MAX);
            trie_id_to_token_id[trie_id] = p.second;
        }
    }
    
    vector<int> get_valid_tokens(const string& remaining_text) {
        vector<int> result;
        trie.predictive_search(remaining_text, [&remaining_text, &result, this](const uint64_t i, const string_view token) {
            if (token.size() > remaining_text.size()) {
                result.push_back(this->trie_id_to_token_id[i]);
            }
        });
        return result;
    }
    
private:
    xcdat::trie_8_type trie;
    vector<int> trie_id_to_token_id;
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
