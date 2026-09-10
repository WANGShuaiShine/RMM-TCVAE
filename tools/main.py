from mmsdk import mmdatasdk

if __name__=='__main__':
    cmumosi_highlevel = mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel,'cmumosi/')
    cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels, 'cmumosi/')

    cmumosi_highlevel.align('glove_vectors', collapse_functions=[myavg])
    cmumosi_highlevel.add_computational_sequences(mmdatasdk.cmu_mosi.labels, 'cmumosi/')
    cmumosi_highlevel.align('Opinion Segment Labels')