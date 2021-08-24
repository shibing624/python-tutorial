import json
import threading

mutex = threading.Lock()


def writetoTxt(json_file, id):
    mutex.acquire()
    data = load_json(json_file)
    print('data', data, type(data))
    data.update({id: {"username": str(id), "sex": 'man'}})
    save_json(data, json_file)
    mutex.release()
    print('new data', data, type(data))


def load_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def save_json(data, json_file):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == '__main__':
    a = {
        "lili": {"username": "lili_01", "sex": "woman", "age": 12, "address": "beijing,china."},
        "lucy": {"username": "lucy_1", "sex": "woman", "age": 32}
    }
    json_file = "a.json"
    save_json(a, json_file)

    for i in range(5):
        myThread = threading.Thread(target=writetoTxt, args=(json_file,i,))
        myThread.start()
