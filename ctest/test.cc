#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <map>

#include <xcdat.hpp>
#include "utils.hpp"

using namespace std;

namespace py = pybind11;

constexpr size_t CACHE_SIZE = 32;

// minimum size to be eligible to be in LRU cache
constexpr int CACHE_THRESHOLD = 100;

class FastTrie {
public:
    FastTrie(const vector<string>& keys) : query_cache(CAChE_SIZE) {
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
    
    py::array_t<int> get_valid_tokens(const string& remaining_text) {
        SharedVector* result;
        if (query_cache.get(remaining_text, result)) {
            // increment reference count
            result->ref++;
        } else {
            result = new SharedVector();
            trie.predictive_search(remaining_text, [&remaining_text, result, this](const uint64_t i, const string_view token) {
                if (token.size() > remaining_text.size()) {
                    result->vec.push_back(this->trie_id_to_token_id[i]);
                }
            });
            // if this was a difficult query, add to cache
            if (result->vec.size() > CACHE_THRESHOLD) {
                // note: query cache increments the ref count internally
                query_cache.set(remaining_text, result);
            }
        }
        return py::array_t<int>(
                {result->vec.size()},
                {sizeof(int)},
                result->vec.data(),
                py::capsule(result, [](void *p) {
                    SharedVector* data = static_cast<SharedVector*>(p);
                    decref(data);
                })
        );
    }
    
private:
    xcdat::trie_8_type trie;
    vector<int> trie_id_to_token_id;
    LRUCache query_cache;
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
