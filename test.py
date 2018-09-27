a = ['sdfsdx','23450d','weruosdf','230dgjsh']
with open('key_list.txt', 'w') as f:
    for key in a:
        f.writelines(key+'\n')

key_list = list()
with open('key_list.txt', 'r') as f:
    b = f.readline().strip()
    while b:
        print(b)
        key_list.append(b)
        b = f.readline().strip()
