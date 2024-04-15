from src.unigram import UnigramTokenizer
from collections import defaultdict
from pyecharts import options as opts
from pyecharts.charts import Tree
from transformers import GPT2Tokenizer


class TreeNode:
    def __init__(self, val, depth, left, right, score, span_score):
        self.val = val
        self.left = left
        self.right = right
        self.depth = depth
        self.score = score
        self.span_score = span_score


class TreeTokenizer:
    def __init__(self, built_vocab, original_tokenizer='gpt2',
                 end_of_token_id=50256, zero_freq_score=17.140061532987676,
                 space_score=50, new_line_id=198):
        self.vocab = built_vocab
        self.tokenizer = GPT2Tokenizer.from_pretrained(original_tokenizer)
        self.ids = []
        self.end_of_token_id = end_of_token_id
        self.zero_freq_score = zero_freq_score
        self.space_score = space_score
        self.new_line_id = new_line_id

    def span_scores(self, s):
        if s == ' ':
            return self.space_score
        if s[0] == ' ':
            s = s[1:]
        if s not in self.vocab:
            return self.zero_freq_score
        else:
            return self.vocab[s]

    def cky_parse(self, sentence):
        n = len(sentence)
        dp = defaultdict(lambda: defaultdict(int))
        bp = defaultdict(TreeNode)

        # Initialize the diagonal entries
        for i in range(n):
            dp[i][i + 1] = self.span_scores(sentence[i:i + 1])
            bp[(i, i + 1)] = TreeNode(sentence[i:i + 1], 1, None, None,
                                      dp[i][i + 1], dp[i][i + 1])

        # Apply the CKY algorithm
        for span in range(2, n + 1):
            for start in range(n - span + 1):
                end = start + span
                for split in range(start + 1, end):
                    score = dp[start][split] + dp[split][
                        end] + self.span_scores(
                        sentence[start:end])
                    if score < dp[start][end] or dp[start][end] == 0.0:
                        dp[start][end] = score
                        bp[(start, end)] = TreeNode(sentence[start:end], span,
                                                    bp[(start, split)],
                                                    bp[(split, end)],
                                                    score, self.span_scores(
                                sentence[start:end]))

        # Find the best parse tree
        best_parse = bp[(0, n)]
        return best_parse, bp

    def compute_node_score(self, node):
        if node is None:
            return TreeNode('None', 0, None, None, self.zero_freq_score,
                            self.zero_freq_score)
        else:
            left = self.compute_node_score(node.left)
            right = self.compute_node_score(node.right)
            node.score = (node.span_score + left.score + right.score) / 3.0
            # print(node.val + str(node.score))
            return node

    def tokens_to_ids(self, generated_tree):
        if generated_tree is None:
            return
        token = generated_tree.val
        token = token.replace(' ', 'Ġ')
        converted_id = self.tokenizer.convert_tokens_to_ids(token)
        if converted_id == self.end_of_token_id:
            self.tokens_to_ids(generated_tree.left)
            self.tokens_to_ids(generated_tree.right)
            return
        if generated_tree.left is None and generated_tree.right is None:
            self.ids.append(converted_id)
            return
        if generated_tree.score < 19.0:
            self.ids.append(converted_id)
            return
        else:
            self.tokens_to_ids(generated_tree.left)
            self.tokens_to_ids(generated_tree.right)
            return

    def encode(self, text):
        split_sentences = text.split('\n')
        self.ids = []
        for sentence in split_sentences:
            parsed_tree, _ = self.cky_parse(sentence)
            parsed_tree = self.compute_node_score(parsed_tree)
            self.tokens_to_ids(parsed_tree)
            self.ids.append(self.new_line_id)
        if len(self.ids) == 0:
            return self.ids
        else:
            return self.ids[:-1]

    def show_tree(self, text):
        split_sentences = text.split('\n')
        sentence = split_sentences[0]
        parsed_tree, _ = self.cky_parse(sentence)
        parsed_tree = self.compute_node_score(parsed_tree)
        return parsed_tree


class PlotTree:
    def __init__(self, orient='TB', title="Parse Tree"):
        self.orient = orient
        self.title = title

    def plot(self, generated_tree, path):
        plot = self.plot_parse_tree(generated_tree)
        plot.render(path)

    def build_tree(self, node):
        if node is None:
            return None
        children = []
        if node.left is not None:
            children.append(self.build_tree(node.left))
        if node.right is not None:
            children.append(self.build_tree(node.right))
        return {"name": node.val + ": " + str(round(node.score, 1)),
                "children": children}

    def plot_parse_tree(self, generated_tree):
        tree_data = self.build_tree(generated_tree)
        c = (
            Tree()
            .add("", [tree_data], collapse_interval=2, orient=self.orient,
                 label_opts=opts.LabelOpts(position="top"), )
            .set_global_opts(title_opts=opts.TitleOpts(title=self.title))
        )
        return c


if __name__ == '__main__':
    # Example usage
    sentences = """def compute_node_score(int): a = b\n axb = math.abs(a,b)"""

    # Build Vocab Freq Score (-log)
    unigram_tokenizer = UnigramTokenizer(corpus_path='text_mag.txt')
    vocab, _ = unigram_tokenizer.init_word_count()

    # Init new Tokenizer
    new_tokenizer = TreeTokenizer(vocab)
    new_ids = new_tokenizer.encode(sentences)
    parse_tree = new_tokenizer.show_tree(sentences)

    # Plot the generated tree
    treeplot = PlotTree()
    treeplot.plot(parse_tree, 'parse_tree.html')

    # Compare with old BPE tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    old_ids = tokenizer.encode(text=sentences, add_special_tokens=True)
    print(old_ids)
    print(new_ids)
