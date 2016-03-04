from VisualOdometry import VisualOdometry
from utils import get_structure, plot_matches, correlate_image, plot_scene
from CVImage import CVImage
from Matcher import Matcher


def run():
    match = Matcher()
    img = CVImage('/home/cesar/Documentos/Computer_Vision/01/image_0')

    # Load images
    img.read_image()
    img.copy_image()
    img.acquire()
    # t = threading.Thread(target=plot_image, args=(img.new_image, ))
    # t.start()

    # Correlate

    correlate_image(match, img, 2, 7)
    print ("Total number of keypoints in second image: \
           {}".format(len(match.global_kpts1)))

    print ("Total number of keypoints in first image: \
           {}".format(len(match.global_kpts2)))

    if not match.is_minmatches:
        print "There aren't matches after filtering. Iterate to next image..."
        return

    # Plot keypoints
    plot_matches(match, img)
    # t.__stop()

    # Now, estimate F
    vo = VisualOdometry()
    match.global_kpts1, match.global_kpts2 = \
        vo.EstimateF_multiprocessing(match.global_kpts2, match.global_kpts1)

    # Get structure of the scene, up to a projectivity
    scene = get_structure(match, img, vo)

    # Plot it
    plot_scene(scene)

if __name__ == '__main__':
    run()
