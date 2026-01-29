# /pkg contains the container package

md5sum_pyx() {
    cat $(find $1 -name "*.pyx" | sort) | md5sum | cut -d" " -f1
}

if [[ $(md5sum_pyx /pkg/src) != $(md5sum_pyx /src/src) ]]; then
    pip install /src/vision_toolkit_c
fi

pip install "/src[test]"

cd /src && python3 tests/hollywood2/run.py
