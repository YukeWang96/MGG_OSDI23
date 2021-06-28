# docker run -it --rm --user yuke_wang --gpus all -v $PWD/../:/MGG mgg:latest /bin/bash
# docker run -it --rm --user yuke_wang --gpus all -v $PWD/../:/MGG mgg:latest /bin/bash
# docker run -it --rm --gpus all -v $PWD/../:/MGG mgg:latest /bin/bash

# user mode
# docker run -it --rm -u $(id -u) --gpus all -v $PWD/../:/MGG mgg:latest /bin/bash

# root mode
docker run -it --rm --gpus all -v $PWD/../:/MGG happy233/mgg:with_ncu /bin/bash