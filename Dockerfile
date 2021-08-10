FROM tensorflow/tensorflow:latest-gpu

WORKDIR /home/krithika/DDH_Project/DDH_Project/

# Install tensorflow and reqs
RUN pip install -U pip
RUN pip install -U tensorflow==2.4.0
RUN pip install -U tensorflow-addons
RUN pip install sklearn
RUN pip install pandas
RUN pip install pytest
RUN pip install matplotlib
RUN pip install flake8
RUN pip install numpy
RUN pip install h5py
RUN pip install datetime
RUN pip install scikit-image
RUN pip install tqdm

RUN apt-get update
RUN apt-get install -y tmux
RUN apt-get install -y nano
RUN apt-get install -y sudo
RUN apt-get install -y task-spooler

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*
