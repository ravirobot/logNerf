# Bruce A. Maxwell
# June 2023
# Converts one or more crt or dng images to 16-bit TIFFs scaled [0, 65535]
#
import sys
import imageio
import raw_utility as rtl

def main(argv):

    if len(argv) < 2:
        print("usage: python %s <image path>" % (argv[0]) )
        return

    for filename in argv[1:]:
        suffix = filename.split(".")[-1].lower()
        if suffix != 'dng' and suffix != 'cr2':
            print("Input image %s is not a supported raw image file")
            return

        print("Processing %s" % (filename) )
        rgb, logrgb = rtl.processRAW( filename )

        # save as a 16-bit TIFF
        words = filename.split(".")
        newfilename = words[0] + ".tif"

        print("Writing %s" % (newfilename) )
        imageio.imsave( newfilename, rgb )

    print("Terminating")

    return

if __name__ == "__main__":
    main(sys.argv)

