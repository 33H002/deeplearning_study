FROM pytorch/pytorch

RUN apt-get -qq update && \
    apt-get install --no-install-recommends -y build-essential \
    curl \
    git \
    vim \
    sudo 

ENV PATH=$PATH:/tmp

ADD requirements.txt /tmp/requirements.txt
ADD startJupyter.sh /tmp/startJupyter.sh

RUN conda install jupyter && \ 
    pip install -r /tmp/requirements.txt && \ 
    chmod +x /tmp/startJupyter.sh