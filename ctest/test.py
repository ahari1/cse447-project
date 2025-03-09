import marisa_ext
import xcdat

trie = marisa_ext.FastTrie(['A', 'c', '-', 'Aa', 'AaA', 'AAa'])

nums = trie.get_valid_tokens('A')

breakpoint()
