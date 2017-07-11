# ------------------------------------------------------------------

#
#   Bayes Optimal Classifier
#
#   In this quiz we will compute the optimal label for a second missing word in a row
#   based on the possible words that could be in the first blank
#
#   Finish the procedurce, LaterWords(), below
#
#   You may want to import your code from the previous programming exercise!
#

sample_memo = '''
Milt, we're gonna need to go ahead and move you downstairs into storage B. We have some new people coming in, and we need all the space we can get. So if you could just go ahead and pack up your stuff and move it down there, that would be terrific, OK?
Oh, and remember: next Friday... is Hawaiian shirt day. So, you know, if you want to, go ahead and wear a Hawaiian shirt and jeans.
Oh, oh, and I almost forgot. Ahh, I'm also gonna need you to go ahead and come in on Sunday, too...
Hello Peter, whats happening? Ummm, I'm gonna need you to go ahead and come in tomorrow. So if you could be here around 9 that would be great, mmmk... oh oh! and I almost forgot ahh, I'm also gonna need you to go ahead and come in on Sunday too, kay. We ahh lost some people this week and ah, we sorta need to play catch up.
'''

corrupted_memo = '''
Yeah, I'm gonna --- you to go ahead --- --- complain about this. Oh, and if you could --- --- and sit at the kids' table, that'd be --- 
'''

data_list = sample_memo.strip().split()

words_to_guess = ["ahead", "could"]


def next_word_probability(sampletext, word):
    words = sampletext.lower().split()
    frequencies = {}
    probabilities = {}

    index = 0

    for item in words:
        if item == word:
            key = words[index + 1]
            frequencies[key] = frequencies.get(key, 0) + 1

        index += 1

    sum_freq = float(sum(frequencies.values()))
    for key, value in frequencies.items():
        probabilities[key] = value / sum_freq

    return frequencies


def later_word_probability(sampletext, probability_dict):

    probabilities = {}

    for word in probability_dict.keys():
        next_word_prob = next_word_probability(sampletext=sampletext, word=word)

        for word2 in next_word_prob.keys():
            probabilities[word2] = probabilities.get(word2, 0) + probability_dict[word] * next_word_prob[word2]

    return probabilities


def LaterWords(sample, word, distance):
    '''@param sample: a sample of text to draw from
    @param word: a word occuring before a corrupted sequence
    @param distance: how many words later to estimate (i.e. 1 for the next word, 2 for the word after that)
    @returns: a single word which is the most likely possibility
    '''

    # TODO: Given a word, collect the relative probabilities of possible following words
    # from @sample. You may want to import your code from the maximum likelihood exercise.
    next_word_prob_dict = next_word_probability(sampletext=sample, word=word)

    # TODO: Repeat the above process--for each distance beyond 1, evaluate the words that
    # might come after each word, and combine them weighting by relative probability
    # into an estimate of what might appear next.
    for i in range(1, distance):
        next_word_prob_dict = later_word_probability(sampletext=sample, probability_dict=next_word_prob_dict)

    prob = {}

    for word, freq in next_word_prob_dict.items():
        prob[word] = freq / float(sum(next_word_prob_dict.values()))

    print prob
    return max(next_word_prob_dict, key=next_word_prob_dict.get)


def test_run():
    # print "My guess is that two blanks after 'ahead' will be: " + LaterWords(sample_memo, "ahead", 2) +\
    #        ". The best guess is 'come'"
    # print "My guess is that two blanks after 'and' will be: " + LaterWords(sample_memo, "and", 2) + \
    #       ". The best guess is 'in'"
    #
    print "You guessed '" + LaterWords(sample_memo, "you", 2) + "' after 'you', where we guessed 'go'"
    print "You guessed '" + LaterWords(sample_memo, "need", 2) + "' after 'need', where we guessed 'to'"
    # print LaterWords(sample_memo, "need", 1)
    # print LaterWords(sample_memo, "you", 1)
    # print LaterWords(sample_memo, "to", 1)

if __name__ == '__main__':
    test_run()
