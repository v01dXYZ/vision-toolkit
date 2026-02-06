# /pkg contains the container package

md5sum_pyx() {
    cat $(find $1 -name "*.pyx" | sort) | md5sum | cut -d" " -f1
}


VISION_TOOLKIT_BUILD="py"

if [[ $(md5sum_pyx /pkg/src) != $(md5sum_pyx /src/src) ]]; then
    VISION_TOOLKIT_BUILD="all"
fi

VISION_TOOLKIT_BUILD=$VISION_TOOLKIT_BUILD pip install "/src[test]" --no-deps

cd /src/tests && $@
