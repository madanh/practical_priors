import os
import shutil
import analyse

tracedir = 'traces'
def getdirlist(dir):
    return sorted(next(os.walk(dir))[1])


def main(tracedir=tracedir):
    # walk the traces folder
    # for each numerical folder run analyse.py with image dir equal to a tracedir/analysis
    for i in getdirlist(tracedir):
        outdir = os.path.join(tracedir,i,'analysis')
        if os.path.exists(outdir):
            print(outdir+' already exists')
        else:
            analyse.main(output_dir=outdir,chains=[str(i)])



def reset(tracedir=tracedir):
    for i in getdirlist(tracedir):
        outdir = os.path.join(tracedir,i,'analysis')
        try:
            shutil.rmtree(outdir)
        except:
            print(outdir+"doesn't exist")
            pass

if __name__=="__main__":
    main()
