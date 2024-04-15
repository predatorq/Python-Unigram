from src.unigram import UnigramTokenizer
from collections import defaultdict
from pyecharts import options as opts
from pyecharts.charts import Tree

tokenizer = UnigramTokenizer(corpus_path='text_mag.txt')
vocab, wordcount = tokenizer.init_word_count()


class TreeNode:
    def __init__(self, val, depth, left, right, score, span_score):
        self.val = val
        self.left = left
        self.right = right
        self.depth = depth
        self.score = score
        self.span_score = span_score


def span_scores(s):
    if s == ' ':
        return 50
    if s[0] == ' ':
        s = s[1:]
    if s not in vocab:
        return 17.140061532987676
    else:
        return vocab[s]


def cky_parse(sentence, span_scores):
    n = len(sentence)
    dp = defaultdict(lambda: defaultdict(int))
    bp = defaultdict(TreeNode)

    # Initialize the diagonal entries
    for i in range(n):
        char = sentence[i]
        dp[i][i + 1] = span_scores(sentence[i:i + 1])
        bp[(i, i + 1)] = TreeNode(sentence[i:i + 1], 1, None, None,
                                  dp[i][i + 1], dp[i][i + 1])

    # Apply the CKY algorithm
    for span in range(2, n + 1):
        for start in range(n - span + 1):
            end = start + span
            for split in range(start + 1, end):
                score = dp[start][split] + dp[split][end] + span_scores(
                    sentence[start:end])
                if score < dp[start][end] or dp[start][end] == 0.0:
                    dp[start][end] = score
                    bp[(start, end)] = TreeNode(sentence[start:end], span,
                                                bp[(start, split)],
                                                bp[(split, end)],
                                                score, span_scores(
                            sentence[start:end]))

    # Find the best parse tree
    best_parse = bp[(0, n)]
    return best_parse, bp


# Example usage
sentence = """def compute_node_score(int): a = b\n axb = math.abs(a,b)"""

parse_tree, _ = cky_parse(sentence, span_scores)


def compute_node_score(node):
    if node is None:
        return TreeNode('None', 0, None, None, 17.140061532987676, 17.140061532987676)
    else:
        left = compute_node_score(node.left)
        right = compute_node_score(node.right)
        node.score = (node.span_score + left.score + right.score) / 3.0
        # print(node.val + str(node.score))
        return node

parse_tree = compute_node_score(parse_tree)

def build_tree(node):
    if node is None:
        return None
    children = []
    if node.left is not None:
        children.append(build_tree(node.left))
    if node.right is not None:
        children.append(build_tree(node.right))
    return {"name": node.val + ": " + str(round(node.score, 1)),
            "children": children}


def plot_parse_tree(parse_tree):
    tree_data = build_tree(parse_tree)
    c = (
        Tree()
        .add("", [tree_data], collapse_interval=2, orient="TB",
             label_opts=opts.LabelOpts(position="top"), )
        .set_global_opts(title_opts=opts.TitleOpts(title="Parse Tree"))
    )
    return c


# Example usage
plot = plot_parse_tree(parse_tree)
plot.render("parse_tree.html")

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
ids = []
def tokens_to_ids(parse_tree):
    if parse_tree is None:
        return
    token = parse_tree.val
    token = token.replace(' ', 'Ġ')
    id = tokenizer.convert_tokens_to_ids(token)
    if id == 50256:
        tokens_to_ids(parse_tree.left)
        tokens_to_ids(parse_tree.right)
        return
    if parse_tree.left is None and parse_tree.right is None:
        ids.append(id)
        return
    if parse_tree.score < 19.0:
        ids.append(id)
        return
    else:
        tokens_to_ids(parse_tree.left)
        tokens_to_ids(parse_tree.right)
        return


tokens_to_ids(parse_tree)
encoded_text = tokenizer.encode(text=sentence, add_special_tokens=True)
print(encoded_text)
print(ids)
