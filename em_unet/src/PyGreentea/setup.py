import os, sys, inspect
import multiprocessing
import platform

def setup_paths(caffe_path, malis_path):
    cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0]))
    if cmd_folder not in sys.path:
        sys.path.append(cmd_folder)
        
    cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], caffe_path + "/python")))
    if cmd_subfolder not in sys.path:
        sys.path.append(cmd_subfolder)
        
    cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], malis_path)))
    if cmd_subfolder not in sys.path:
        sys.path.append(cmd_subfolder)
    
    sys.path.append(caffe_path + "/python")
    sys.path.append(malis_path + "/python")


def linux_distribution():
    try:
        return platform.linux_distribution()
    except:
        return "N/A"


def sys_info():
    print("""Python version: %s
    dist: %s
    linux_distribution: %s
    system: %s
    machine: %s
    platform: %s
    uname: %s
    version: %s
    mac_ver: %s
    """ % (
           sys.version.split('\n'),
           str(platform.dist()),
           linux_distribution(),
           platform.system(),
           platform.machine(),
           platform.platform(),
           platform.uname(),
           platform.version(),
           platform.mac_ver(),
           ))


def install_dependencies():
    # We support Fedora (22/23/24) and Ubuntu (14.05/15.05)
    if (linux_distribution()[0].lower() == "fedora"):
        # TODO: Add missing Fedora packages
        os.system('dnf install -y git gcc')
        os.system('dnf install -y protobuf-python protobuf-c protobuf-compiler')
        os.system('dnf install -y boost-system boost-devel boost-python')
        os.system('dnf install -y glog glog-devel gflags gflags-devel')
        os.system('dnf install -y python python-devel python-pip')
        os.system('dnf install -y atlas atlas-sse2 atlas-sse3')
        os.system('dnf install -y openblas openblas-devel openblas-openmp64 openblas-openmp openblas-threads64 openblas-threads')
        os.system('dnf install -y opencl-headers')
    if (linux_distribution()[0].lower() == "ubuntu"):
        # TODO: Add missing Ubuntu packages
        os.system('apt-get update -y')
        os.system('apt-get install -y git gcc')
        os.system('apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev')
        os.system('apt-get install -y protobuf-compiler gfortran libjpeg62 libfreeimage-dev libatlas-base-dev')
        os.system('apt-get install -y libopenblas-base libopenblas-dev')
        os.system('apt-get install -y libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev')
        os.system('apt-get install -y python-dev python-pip python-yaml')
        os.system('apt-get install -y libviennacl-dev opencl-headers')
    
    os.system('pip install --upgrade pip')
    os.system('pip install cython')
   
def compile_malis(path):
    cwd = os.getcwd()
    os.chdir(path)
    os.system('sh make.sh')
    os.chdir(cwd)

def compile_caffe(path):
    cpus = multiprocessing.cpu_count()
    cwd = os.getcwd()
    os.chdir(path)
    # Copy the default Caffe configuration if not existing
    os.system("cp -n Makefile.config.example Makefile.config")
    result = os.system("make all -j %s" % cpus)
    if result != 0:
        sys.exit(result)
    result = os.system("make pycaffe -j %s" % cpus)
    if result != 0:
        sys.exit(result)
    os.chdir(cwd)
    
def clone_malis(path, clone, update):
    if clone:
        os.system('git clone https://github.com/srinituraga/malis.git %s' % path)
    if update:
        cwd = os.getcwd()
        os.chdir(path)
        os.system('git pull')
        os.chdir(cwd)

def clone_caffe(path, clone, update):
    if clone:
        os.system('git clone https://github.com/naibaf7/caffe.git %s' % path)
    if update:
        cwd = os.getcwd()
        os.chdir(path)
        os.system('git pull')
        os.chdir(cwd)
        

def set_environment_vars():
    # Fix up OpenCL variables. Can interfere with the
    # frame buffer if the GPU is also a display driver
    os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
    os.environ["GPU_SINGLE_ALLOC_PERCENT"] = "100"
    os.environ["GPU_MAX_HEAP_SIZE"] = "100"
    os.environ["GPU_FORCE_64BIT_PTR"] = "1"
