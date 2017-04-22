import numpy as np
import time
import os
import pdb

def rankpooling(models, *Oc_only): # Oc_only: Occupied layer only ((keywork argument?
    if Oc_only[0] == None:
        N = models.shape[2]
        fw = []
        for i in xrange(0+1,N+1):
            fw.append(sum(((2*np.arange(i,N+1).astype(float)-N-1) / np.arange(i,N+1).astype(float)))) #after column 16, all zeros
        new_fw = np.asarray(fw).reshape([1,1,N])
        model = np.sum(models*new_fw, axis=2)
    elif Oc_only[0] == True:
        N = 0 
        whichN = []
        for layer in xrange(models.shape[2]):
            occ = np.where(models[:,:,layer]>0)
            if len(occ[0]) and len(occ[1]):
                whichN.append(layer)
                N += 1
        fw = []
        for i in xrange(len(whichN)):#xrange(0+1,N+1):
            fw.append(sum(((2*np.arange(int(i)+1,N+1).astype(float)-N-1) / np.arange(int(i)+1,N+1).astype(float)))) #after column 16, all zeros
        pdb.set_trace()
        new_fw = np.asarray(fw).reshape([1,1,N])
        model = np.sum(models*new_fw, axis=2)
    return model

def get_template_path(toread):
    toread_path = os.path.join(os.getcwd(), toread)
    if not os.path.isfile(toread_path):
        raise Exception("This is not a valid path: %s"%(toread_path))
    return toread_path

def get_template(toread, option):
    toread_path = get_template_path(toread)
    if option.endswith('train_dir'):
        return open(toread_path).read().splitlines()[0]
    elif option.endswith('test_dir'):
        return open(toread_path).read().splitlines()[1]


def render_context(template_string, context):
    return template_string.format(**context)

if __name__ == '__main__':
    start_time = time.time()
    ### main script
    if not os.path.exists('./train_dir'):
        os.makedirs('./train_dir')
    if not os.path.exists('./test_dir'):
        os.makedirs('./test_dir')

    source_path = ['../../voxnet/scripts/train_dir', '../../voxnet/scripts/test_dir']
    classnames = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']

    for path in source_path:
        for classname in classnames:
            for fname in os.listdir(path):
                if fname.startswith(classname):
                    if classname == 'night_stand':
                        instance = fname.split('_')[2]
                        rot = fname.split('_')[3].split('.')[0]
                    else:
                        instance = fname.split('_')[1]
                        rot = fname.split('_')[2].split('.')[0]
                    toread_ = 'path.txt'
                    template = get_template(toread_, path)
                    context = {
                            'classname' : '%s'%(classname),
                            'instance' : '%09d'%(int(instance)),
                            'rot' : '%02d' %(int(rot))
                            }
                    #file format: 'sofa_000000585_08.npy'
                    data_models = np.load(render_context(template, context))
                    data_model = rankpooling(data_models, True)
                    if path.endswith('train_dir'):
                        #np.savez('train_dir/'+fname.split('.')[0], classname=classname, data=data_model)
                        print(fname.split('.')[0])
                    elif path.endswith('test_dir'):    
                        #np.savez('test_dir/'+fname.split('.')[0], classname=classname, data=data_model)
                        print(fname.split('.')[0])
    elapsed_time = time.time() - start_time
    print('total elapsed time:'+str(elapsed_time))
