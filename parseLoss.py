import pdb

def loadfile(filename):
    with open(filename, 'r') as infile:
        line = ''
        iter_dict = {}
        while True:
            while 'Iteration' not in line:
                try:
                    line = next(infile)
                    continue
                except:
                    return iter_dict
            entry = {}
            set_index = 5
            iteration = line.split(' ')[set_index].split(',')[0] #5-->6
            string = line.split(' ')[set_index+1] #6-->7
            print iteration
            print string
            try:
                if (int(iteration) % 40 == 0) and (string == 'loss'):
                    loss = line.split(' ')[set_index+3].split('\n')[0] #8 --> 9
                    loss = float(loss)
                    print loss
                    if loss > 3.0:
                        loss = 3.0
                    iter_dict[iteration] = loss
            except ValueError:
                print 'value error'
            line = next(infile)
