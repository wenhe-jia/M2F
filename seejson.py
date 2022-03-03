import json


def main():
    jsonfile = json.load(open('./datasets/ytvis_2021/train.json', 'rb'))
    print(jsonfile['annotations'][0])


if __name__ == '__main__':
    main()
