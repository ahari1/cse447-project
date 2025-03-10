#include <unordered_map>
#include <list>
#include <vector>
#include <cassert>
#include <utility>

// This serves as an object where we can reference count our vectors
struct SharedVector
{
    std::vector<int> vec;
    int ref = 1;
};

inline void decref(SharedVector* sv) {
    sv->ref--;
    if (sv->ref == 0) {
        delete sv;
    }
}

// Copied from https://stackoverflow.com/a/54272232
class LRUCache
{

private:
    std::list<std::string>items;
    std::unordered_map <std::string, std::pair<SharedVector *const, typename std::list<std::string>::iterator>> keyValuesMap;
    size_t csize;

public:
    LRUCache(int s) :csize(s) {
        if (csize < 1)
            csize = 32;
    }

    void set(const std::string& key, SharedVector *const value) {
        value->ref++;
        items.push_front(key);
        keyValuesMap.emplace(key, std::make_pair(value, items.begin()));
        if (keyValuesMap.size() > csize) {
            // we also decrement the reference count
            auto pos2 = keyValuesMap.find(items.back());
            assert(pos2 != keyValuesMap.end()); 
            decref(pos2->second.first);
            keyValuesMap.erase(pos2);
            items.pop_back();
        }
    }

    bool get(const std::string& key, SharedVector*& value) {
        auto pos = keyValuesMap.find(key);
        if (pos == keyValuesMap.end())
            return false;
        items.erase(pos->second.second);
        items.push_front(key);
        pos->second.second = items.begin();
        value = pos->second.first;
        return true;
    }
};

