import argparse
import sys
import util
sys.path.append('skip-thoughts')
import skipthoughts


def main(params):
    captions_file = params.captions_file
    output_file = params.output_file

    vids = util.load_pkl(captions_file)
    st_model = skipthoughts.load_model()

    skip_vectors = {}
    for vid in vids.keys():

        caps = vids[vid]
        num_caps = len(caps)

        raw_caps = [ '' for x in range(num_caps)]

        for cap in caps:
            raw_caps[int(cap['cap_id'])]=cap['tokenized']

        vector = skipthoughts.encode(st_model, raw_caps, verbose=False)

        skip_vectors[vid] = vector

    util.dump_pkl(skip_vectors, output_file)


if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-i','--input',dest ='captions_file',type=str, required=True)
    arg_parser.add_argument('-o','--output',dest ='output_file',type=str, required=True, help="/path/to/dataset/skip_vectors.pkl")

    args = arg_parser.parse_args()

    main(args)
