def count_lines(file_path):
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
         while True:
            line = file.readline()
            if not line:
                break
            print(line)
            count += 1
    return count

if __name__ == "__main__":
    print(count_lines("brwac/brwac_plain.txt"))