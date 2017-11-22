data_root = 'train'
label_root = 'data/'
category_file = open(label_root + 'categories.txt', 'r')
label_file = open(label_root + 'temp.txt', 'w')

categories = category_file.readlines()

for c in categories:
    sc = c.split(" ")
    for i in range(1, 1001):
        label_file.write(data_root + sc[0] + "/" + "%08d" % (i,)+ ".jpg " + sc[1])
    for j in range(2001, 5001):
        label_file.write(data_root + sc[0] + "/" + "%08d" % (j,)+ ".jpg " + sc[1])

label_file.close()
category_file.close()
