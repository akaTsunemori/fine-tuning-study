import sys


def main() -> None:
    args = sys.argv[1:]
    if '--epochs' in args:
        pass
    if '--model' in args:
        pass
    print(args)


if __name__ == '__main__':
    main()