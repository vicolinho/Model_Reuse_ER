
def percentage_distance(val1, val2):
    if val1 != '' and val2 != '':
        min_v = min(val1, val2)
        max_v = max(val1, val2)
        if min_v == max_v:
            return 1
        else:
            return min_v/max_v
    else:
        return 0
