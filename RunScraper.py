from scraper import Scraper


def main():
    target_file = 'targets.txt'

    lines = [line[:-1] for line in open(target_file, 'r', encoding='utf-8')]
    split = [line.split('\t') for line in lines if
             not (line.startswith('#') or len(line) == 0)]
    targets = [(line[0], line[1]) for line in split]

    for t in targets:
        Scraper.save_stock_data(t[0], t[1])


if __name__ == '__main__':
    main()
