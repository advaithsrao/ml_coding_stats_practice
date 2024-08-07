import copy

original = [0,1,2,3,4]
shallow_copy = original
deep_copy = copy.deepcopy(original)

print('BEFORE')
print('original:', original)
print('shallow:', shallow_copy)
print('deep:', deep_copy)

original[2] = -2

print('AFTER ')
print('original:', original)
print('shallow:', shallow_copy)
print('deep:', deep_copy)
