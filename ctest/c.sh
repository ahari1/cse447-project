g++ -Wall -shared -std=c++17 -fPIC -Ofast \
    $(python3-config --includes) \
    $(python3 -m pybind11 --includes) \
    test.cc \
    -o marisa_ext$(python3-config --extension-suffix) \
    $(python3-config --ldflags) \
    $(python3-config --libs) \
    -Ixcdat/include

# g++ -Wall -shared -std=c++11 -fPIC \
#     $(python3-config --includes) \
#     $(python3 -m pybind11 --includes) \
#     test-marisa.cc \
#     -Imarisa-trie/include \
#     -o marisa_ext$(python3-config --extension-suffix) \
#     $(python3-config --ldflags) \
#     $(python3-config --libs) \
#     -Lmarisa-trie/
    # -L/opt/anaconda3/envs/pybind/lib \
    # -lpython3.11
