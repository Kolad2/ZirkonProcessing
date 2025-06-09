import yadisk
import json

#pip install yadisk


def main():
    with open("./config.json") as file:
        config = json.load(file)

    public_link = "https://disk.yandex.ru/d/5olsVShHNEzPgA"

    token = config["token"]
    y = yadisk.YaDisk(token=token)

    if y.check_token():
        print(True)
    else:
        print("token is not valid")



    public_resource = y.get_public_meta(public_link)
    if public_resource.type == "dir":
        print("я папка")
        print(public_resource.name)
        for item in y.listdir(public_resource.path):
            print(item.path)
    #print(public_resource)


if __name__ == '__main__':
    main()