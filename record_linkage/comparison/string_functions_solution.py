""" Module consisting of similarity functions for  comparing two strings or precomputed token sets
"""

Q = 2  # Value length of q-grams for Jaccard and Dice comparison function
is_efficient = False
is_padding = True


# =============================================================================
# First the basic functions to compare attribute values

def exact_comp(val1, val2):
    """Compare the two given attribute values exactly, return 1 if they are the
     same (but not both empty!) and 0 otherwise.
  """

    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    elif val1 != val2:
        return 0.0
    else:  # The values are the same
        return 1.0


# -----------------------------------------------------------------------------

def jaccard_comp(val1, val2):
    """Calculate the Jaccard similarity between the two given attribute values
     by extracting sets of sub-strings (q-grams) of length q.

     Returns a value between 0.0 and 1.0.
  """
    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif val1 == val2:
        return 1.0

    # ********* Implement Jaccard similarity function here *********

    jacc_sim = 0.0  # Replace with your code
    if not is_efficient:
        if is_padding:
            pad_val_1 = "#" * (Q - 1) + val1 + "#" * (Q - 1)
            pad_val_2 = "#" * (Q - 1) + val2 + "#" * (Q - 1)
        else:
            pad_val_1 = val1
            pad_val_2 = val2
        q_gram_list1 = [pad_val_1[i:i + Q] for i in range(len(pad_val_1) - (Q - 1))]
        q_gram_list2 = [pad_val_2[i:i + Q] for i in range(len(pad_val_2) - (Q - 1))]

        q_gram_set1 = set(q_gram_list1)
        q_gram_set2 = set(q_gram_list2)
    else:
        q_gram_set1 = val1
        q_gram_set2 = val2

    i = len(q_gram_set1.intersection(q_gram_set2))
    u = len(q_gram_set1.union(q_gram_set2))
    assert u > 0, u

    jacc_sim = float(i) / u

    # ************ End of your Jaccard code *************************************

    assert jacc_sim >= 0.0 and jacc_sim <= 1.0

    return jacc_sim


# -----------------------------------------------------------------------------

def dice_comp(val1, val2):
    """Calculate the Dice coefficient similarity between the two given attribute
     values by extracting sets of sub-strings (q-grams) of length q.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif val1 == val2:
        return 1.0
    if not is_efficient:
        if is_padding:
            pad_val_1 = "#" * (Q - 1) + val1 + "#" * (Q - 1)
            pad_val_2 = "#" * (Q - 1) + val2 + "#" * (Q - 1)
        else:
            pad_val_1 = val1
            pad_val_2 = val2
        q_gram_list1 = [pad_val_1[i:i + Q] for i in range(len(pad_val_1) - (Q - 1))]
        q_gram_list2 = [pad_val_2[i:i + Q] for i in range(len(pad_val_2) - (Q - 1))]
        q_gram_set1 = set(q_gram_list1)
        q_gram_set2 = set(q_gram_list2)
    else:
        q_gram_set1 = val1
        q_gram_set2 = val2
    if len(q_gram_set1) == 0 or len(q_gram_set2) == 0:
        return 0
    i = len(q_gram_set1.intersection(q_gram_set2))
    dice_sim = 2.0 * float(i) / float(len(q_gram_set1) + len(q_gram_set2))

    # ************ End of your Dice code ****************************************

    assert 0.0 <= dice_sim <= 1.0

    return dice_sim


# -----------------------------------------------------------------------------

JARO_MARKER_CHAR = chr(1)  # Special character used in the Jaro, Winkler comp.


def jaro_comp(val1, val2):
    """Calculate the similarity between the two given attribute values based on
     the Jaro comparison function.

     As described in 'An Application of the Fellegi-Sunter Model of Record
     Linkage to the 1990 U.S. Decennial Census' by William E. Winkler and Yves
     Thibaudeau.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (val1 == '') or (val2 == ''):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    len1 = len(val1)  # Number of characters in val1
    len2 = len(val2)  # Number of characters in val2

    halflen = int(max(len1, len2) / 2) - 1

    assingment1 = ''  # Characters assigned in val1
    assingment2 = ''  # Characters assigned in val2

    workstr1 = val1  # Copy of original value1
    workstr2 = val2  # Copy of original value1

    common1 = 0  # Number of common characters
    common2 = 0  # Number of common characters

    for i in range(len1):  # Analyse the first string
        start = max(0, i - halflen)
        end = min(i + halflen + 1, len2)
        index = workstr2.find(val1[i], start, end)
        if index > -1:  # Found common character, count and mark it as assigned
            common1 += 1
            assingment1 = assingment1 + val1[i]
            workstr2 = workstr2[:index] + JARO_MARKER_CHAR + workstr2[index + 1:]

    for i in range(len2):  # Analyse the second string
        start = max(0, i - halflen)
        end = min(i + halflen + 1, len1)
        index = workstr1.find(val2[i], start, end)
        if index > -1:  # Found common character, count and mark it as assigned
            common2 += 1
            assingment2 = assingment2 + val2[i]
            workstr1 = workstr1[:index] + JARO_MARKER_CHAR + workstr1[index + 1:]
    if common1 != common2:
        common1 = float(common1 + common2) / 2.0

    if common1 == 0:  # No common characters within half length of strings
        return 0.0

    transposition = 0  # Calculate number of transpositions
    for i in range(len(assingment1)):
        if assingment1[i] != assingment2[i]:
            transposition += 1
    transposition = transposition / 2.0
    common1 = float(common1)

    jaro_sim = 1. / 3. * (common1 / float(len1) + common1 / float(len2) +
                          (common1 - transposition) / common1)

    assert (jaro_sim >= 0.0) and (jaro_sim <= 1.0), \
        'Similarity weight outside 0-1: %f' % (jaro_sim)

    return jaro_sim


# -----------------------------------------------------------------------------

def jaro_winkler_comp(val1, val2):
    """Calculate the similarity between the two given attribute values based on
     the Jaro-Winkler modifications.

     Applies the Winkler modification if the beginning of the two strings is
     the same.

     As described in 'An Application of the Fellegi-Sunter Model of Record
     Linkage to the 1990 U.S. Decennial Census' by William E. Winkler and Yves
     Thibaudeau.

     If the beginning of the two strings (up to first four characters) are the
     same, the similarity weight will be increased.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (val1 == '') or (val2 == ''):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif val1 == val2:
        return 1.0

    # First calculate the basic Jaro similarity
    #
    jaro_sim = jaro_comp(val1, val2)
    if jaro_sim == 0:
        return 0.0  # No common characters
    minlen = min(len(val1), len(val2))

    for c in range(1, minlen + 1):
        if val1[:c] != val2[:c]:
            break
    c -= 1

    if c > 4:
        c = 4

    jw_sim = jaro_sim + c * 0.1 * (1.0 - jaro_sim)

    assert (jw_sim >= jaro_sim), 'Winkler modification is negative'
    assert (jw_sim >= 0.0) and (jw_sim <= 1.0), \
        'Similarity weight outside 0-1: %f' % (jw_sim)

    return jw_sim


# -----------------------------------------------------------------------------

def bag_dist_sim_comp(val1, val2):
    """Calculate the bag distance similarity between the two given attribute
     values.

     Returns a value between 0.0 and 1.0.
  """

    # If at least one of the values is empty return 0
    #
    if (len(val1) == 0) or (len(val2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif (val1 == val2):
        return 1.0

    # ********* Implement bag similarity function here *********
    n = len(val1)
    m = len(val2)

    list1 = list(val1)
    list2 = list(val2)

    for ch in val1:
        if ch in list2:
            list2.remove(ch)

    for ch in val2:
        if ch in list1:
            list1.remove(ch)

    b = max(len(list1), len(list2))

    bag_sim = 1.0 - float(b) / float(max(n, m))

    # ************ End of your bag distance code ********************************

    assert bag_sim >= 0.0 and bag_sim <= 1.0

    return bag_sim


# -----------------------------------------------------------------------------

def edit_dist_sim_comp(s1, s2):
    """Calculate the edit distance similarity between the two given attribute
     values.

     Returns a value between 0.0 and 1.0.
  """
    # If at least one of the values is empty return 0
    #
    if (len(s1) == 0) or (len(s2) == 0):
        return 0.0

    # If both attribute values exactly match return 1
    #
    elif s1 == s2:
        return 1.0

    # ********* Implement edit distance similarity here *********

    edit_sim = 0.0  # Replace with your code

    n = len(s1)
    m = len(s2)

    d = []  # Table with the full distance matrix
    current = range(n + 1)

    d.append(current)

    for i in range(1, m + 1):

        previous = current
        current = [i] + n * [0]
        str2char = s2[i - 1]

        for j in range(1, n + 1):
            substitute = previous[j - 1]
            if s1[j - 1] != str2char:
                substitute += 1

            # Get minimum of insert, delete and substitute
            #
            current[j] = min(previous[j] + 1, current[j - 1] + 1, substitute)

        d.append(current)
    ed = d[m][n]  # Start with final position in table
    edit_sim = 1.0 - float(ed) / float(max(n, m))
    assert 0.0 <= edit_sim <= 1.0
    return edit_sim

if __name__ == '__main__':
    is_efficient = False
    is_padding = False
    sims = [jaro_winkler_comp, edit_dist_sim_comp]
    values = [('jones', 'johnson'),('michelle', 'michael'),('shackleford', 'shackelford'),('mohamed','demahom')]
    for s in sims:
        for t in values:
            sim = s(t[0], t[1])
            print(str(s) +"(" +str(t)+"):"+str(sim))

