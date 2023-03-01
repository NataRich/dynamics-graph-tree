def bruteforce_search(string: str):
    length = len(string)
    begin_index = 0
    while begin_index < length:
        end_index = begin_index + 1
        while end_index <= length:
            substr = string[begin_index:end_index]
            if is_periodic(substr, string[end_index:]):
                return string[:begin_index], substr
            end_index += 1
        begin_index += 1
    return string, ''


def is_periodic(substr: str, rem: str):
    k = len(substr)
    r = len(rem)
    if k == 0 or r == 0 or r < k:
        return False

    for i in range(r):
        if rem[i] != substr[i % k]:
            return False
    return True
