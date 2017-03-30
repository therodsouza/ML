"""Count words."""


def count_words(s, n):
    """Return the n most frequently occuring words in s."""

    # TODO: Count the number of occurences of each word in s
    words = s.lower().split()
    dict = {}

    for item in words:
        dict[item] = words.count(item)

    # TODO: Sort the occurences in descending order (alphabetically in case of ties)
    items = dict.items()

    items.sort(key=lambda tup: tup[0])
    items.sort(key=lambda tup: tup[1], reverse=True)

    # TODO: Return the top n words as a list of tuples (<word>, <count>)
    return items[:n]


def test_run():
    """Test count_words() with some inputs."""
    print count_words("cat bat mat cat bat cat", 3)
    print count_words("betty bought a bit of butter but the butter was bitter", 3)


if __name__ == '__main__':
    test_run()
