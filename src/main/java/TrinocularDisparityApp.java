import boofcv.abst.disparity.ConfigSpeckleFilter;
import boofcv.abst.disparity.DisparitySmoother;
import boofcv.abst.disparity.StereoDisparity;
import boofcv.alg.distort.ImageDistort;
import boofcv.alg.filter.misc.AverageDownSampleOps;
import boofcv.alg.geo.PerspectiveOps;
import boofcv.alg.geo.RectifyDistortImageOps;
import boofcv.alg.geo.RectifyImageOps;
import boofcv.alg.geo.rectify.DisparityParameters;
import boofcv.alg.geo.rectify.RectifyCalibrated;
import boofcv.alg.meshing.DepthImageToMeshGridSample;
import boofcv.alg.meshing.VertexMesh;
import boofcv.factory.disparity.ConfigDisparityBMBest5;
import boofcv.factory.disparity.DisparityError;
import boofcv.factory.disparity.FactoryStereoDisparity;
import boofcv.io.UtilIO;
import boofcv.io.calibration.CalibrationIO;
import boofcv.io.points.PointCloudIO;
import boofcv.misc.BoofMiscOps;
import boofcv.struct.border.BorderType;
import boofcv.struct.calib.CameraPinholeBrown;
import boofcv.struct.calib.MultiCameraCalibParams;
import boofcv.struct.calib.StereoParameters;
import boofcv.struct.image.GrayF32;
import georegression.struct.point.Point2D_F64;
import georegression.struct.se.Se3_F64;
import org.ddogleg.struct.DogArray;
import org.ddogleg.struct.DogArray_I32;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.FMatrixRMaj;
import org.ejml.ops.ConvertMatrixData;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

import static boofcv.io.image.UtilImageIO.loadImage;
import static java.util.Objects.requireNonNull;

/**
 * @author Peter Abeles
 */
public class TrinocularDisparityApp {
    @Option(name = "-i", aliases = {"--Input"}, usage = "Root data directory")
    public String pathInput = "";

    @Option(name = "-c", aliases = {"--Calibration"}, usage = "Location of multiview calibration")
    public String pathCalibration;

    @Option(name = "-o", aliases = {"--Output"}, usage = "Path to output directory")
    public String pathOutput = "trinocular_output";

    @Option(name = "--Left", usage = "Glob pattern for left images inside root data directory")
    public String globLeft = "left*";

    @Option(name = "--Right", usage = "Glob pattern for right images inside root data directory")
    public String globRight = "right*";

    @Option(name = "--Middle", usage = "Glob pattern for middle images inside root data directory")
    public String globMiddle = "middle*";

    @Option(name = "--MaxPixel", usage = "Scales input image so that the total number of pixels match this")
    public int maxPixels = 1024 * 768;

    @Option(name = "--Display", usage = "Displays the 3D cloud in a window")
    public boolean display = false;

    @Option(name = "--DisparityMin", usage = "Minimum disparity value it will consider")
    public int disparityMin = 5;

    @Option(name = "--RegionSize", usage = "Size of blocks it matches when finding disparity")
    public int regionSize = 5;

    // Maximum disparity it will consider. Can't go beyond 255
    public int disparityRange = 255;

    RectifyCalibrated rectifyAlg = RectifyImageOps.createCalibrated();
    DisparityParameters dparam = new DisparityParameters();

    public void process() {
        MultiCameraCalibParams calibration = CalibrationIO.load(pathCalibration);
        MultiCameraCalibParams calibrationScaled = scaleCalibration(calibration);

        List<String> filesLeft = UtilIO.listSmartImages(String.format("glob:%s/%s", pathInput, globLeft), true);
        List<String> filesRight = UtilIO.listSmartImages(String.format("glob:%s/%s", pathInput, globRight), true);
        List<String> filesMiddle = UtilIO.listSmartImages(String.format("glob:%s/%s", pathInput, globMiddle), true);

        BoofMiscOps.checkEq(filesLeft.size(), filesRight.size(), "left + right images miss match");
        BoofMiscOps.checkEq(filesLeft.size(), filesMiddle.size(), "left + middle images miss match\"");

        if (filesLeft.isEmpty()) {
            System.err.println("No images found. path=" + pathInput);
            System.exit(1);
        }

        int numFrames = filesLeft.size();

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++) {
            System.out.println("frame " + frameIdx + " / " + numFrames);
            // Load images. For simplicity, left=0, middle=1, right=2
            GrayF32 image0 = requireNonNull(loadImage(filesLeft.get(frameIdx), GrayF32.class));
            GrayF32 image1 = requireNonNull(loadImage(filesMiddle.get(frameIdx), GrayF32.class));
            GrayF32 image2 = requireNonNull(loadImage(filesRight.get(frameIdx), GrayF32.class));

            // Scale images down;
            GrayF32 scale0 = scaleDown(image0);
            GrayF32 scale1 = scaleDown(image1);
            GrayF32 scale2 = scaleDown(image2);

            // Sanity check
            // This should catch mixing of input images and incorrectly scaled images in this code
            BoofMiscOps.checkTrue(calibrationScaled.intrinsics.get(0).isSameShape(scale0));
            BoofMiscOps.checkTrue(calibrationScaled.intrinsics.get(1).isSameShape(scale1));
            BoofMiscOps.checkTrue(calibrationScaled.intrinsics.get(2).isSameShape(scale2));

            computeStereo(frameIdx, 0, 1, calibrationScaled, scale0, scale1, scale2);
            computeStereo(frameIdx, 0, 2, calibrationScaled, scale0, scale1, scale2);
            computeStereo(frameIdx, 1, 2, calibrationScaled, scale0, scale1, scale2);
        }
    }

    private void computeStereo(int frameID, int camA, int camB, MultiCameraCalibParams calibration, GrayF32... images) {
        // Convert into a stereo pair
        var stereoParam = new StereoParameters();
        calibration.computeExtrinsics(camB, camA, stereoParam.right_to_left);
        stereoParam.left.setTo((CameraPinholeBrown) calibration.intrinsics.get(camA));
        stereoParam.right.setTo((CameraPinholeBrown) calibration.intrinsics.get(camB));

        // storage for rectified images
        var rectLeft = new GrayF32(1, 1);
        var rectRight = new GrayF32(1, 1);
        var disparity = new GrayF32(1, 1);

        rectify(images[camA], images[camB], stereoParam, rectLeft, rectRight);

        denseDisparitySubpixel(rectLeft, rectRight, regionSize, disparityMin, disparityRange,
                disparity);

        // Here's what we came here for. Time to remove the speckle
        var configSpeckle = new ConfigSpeckleFilter();
        // Two pixels are connected if their disparity is this similar
        configSpeckle.similarTol = 2.0f;
        // probably the most important parameter, speckle size
        configSpeckle.maximumArea.setRelative(0.0001, 20);
        DisparitySmoother<GrayF32, GrayF32> smoother =
                FactoryStereoDisparity.removeSpeckle(configSpeckle, GrayF32.class);

        smoother.process(rectLeft, disparity, disparityRange);

        saveMesh(new File(pathOutput, String.format("frame%03d_%dx%d", frameID, camA, camB)), rectLeft, disparity);
    }

    /**
     * Returns a scaled down image or the original if no scaling is needed
     */
    private GrayF32 scaleDown(GrayF32 image) {
        // See if the image is smaller than the target
        int foundPixels = image.width * image.height;
        if (foundPixels <= maxPixels)
            return image;

        // Scale using average down sampling to reduce artifacts
        double scale = Math.sqrt(maxPixels) / Math.sqrt(foundPixels);
        GrayF32 scaled = new GrayF32((int) (image.width * scale), (int) (image.height * scale));
        AverageDownSampleOps.down(image, scaled);
        return scaled;
    }

    /**
     * Recomputes camera calibration after scaling the images
     */
    private MultiCameraCalibParams scaleCalibration(MultiCameraCalibParams original) {
        var scaled = new MultiCameraCalibParams();

        for (int camIdx = 0; camIdx < original.intrinsics.size(); camIdx++) {
            CameraPinholeBrown o = original.getIntrinsics(camIdx);
            CameraPinholeBrown s = new CameraPinholeBrown().setTo(o);

            double scale = Math.sqrt(maxPixels) / Math.sqrt(o.width * o.height);
            PerspectiveOps.scaleIntrinsic(s, scale);

            scaled.intrinsics.add(s);
            scaled.camerasToSensor.add(original.camerasToSensor.get(camIdx));
        }

        return scaled;
    }

    private void saveMesh(File dirOutput, GrayF32 rectLeft, GrayF32 disparity) {
        var alg = new DepthImageToMeshGridSample();
        alg.samplePeriod.setFixed(2);
        alg.processDisparity(dparam, disparity, 2.0f);
        VertexMesh mesh = alg.getMesh();

        // Specify the color of each vertex
        var colors = new DogArray_I32(mesh.vertexes.size());
        DogArray<Point2D_F64> pixels = alg.getVertexPixels();
        for (int i = 0; i < pixels.size; i++) {
            Point2D_F64 p = pixels.get(i);
            int v = (int) rectLeft.get((int) p.x, (int) p.y);
            colors.add(v << 16 | v << 8 | v);
        }

        // Make sure the output directory exists
        UtilIO.mkdirs(dirOutput);

        try (var out = new FileOutputStream(new File(dirOutput, "mesh.ply"))) {
            PointCloudIO.save3D(PointCloudIO.Format.PLY, mesh, colors, out);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void denseDisparitySubpixel(GrayF32 rectLeft, GrayF32 rectRight,
                                              int regionSize,
                                              int disparityMin, int disparityRange,
                                              GrayF32 disparity) {
        // A slower but more accuracy algorithm is selected
        // All of these parameters should be turned
        var config = new ConfigDisparityBMBest5();
        config.errorType = DisparityError.CENSUS;
        config.disparityMin = disparityMin;
        config.disparityRange = disparityRange;
        config.subpixel = true;
        config.regionRadiusX = config.regionRadiusY = regionSize;
//        config.maxPerPixelError = 35;
        config.validateRtoL = 1;
        config.texture = 0.05;
        StereoDisparity<GrayF32, GrayF32> disparityAlg =
                FactoryStereoDisparity.blockMatchBest5(config, GrayF32.class, GrayF32.class);

        // process and return the results
        disparityAlg.process(rectLeft, rectRight);

        disparity.setTo(disparityAlg.getDisparity());
    }

    /**
     * Rectified the input images using known calibration.
     */
    public RectifyCalibrated rectify(GrayF32 origLeft, GrayF32 origRight,
                                     StereoParameters param,
                                     GrayF32 rectLeft, GrayF32 rectRight) {
        rectLeft.reshapeTo(origLeft);
        rectRight.reshapeTo(origRight);

        // Compute rectification
        Se3_F64 leftToRight = param.getRightToLeft().invert(null);

        // original camera calibration matrices
        DMatrixRMaj K1 = PerspectiveOps.pinholeToMatrix(param.getLeft(), (DMatrixRMaj) null);
        DMatrixRMaj K2 = PerspectiveOps.pinholeToMatrix(param.getRight(), (DMatrixRMaj) null);

        rectifyAlg.process(K1, new Se3_F64(), K2, leftToRight);

        // rectification matrix for each image
        DMatrixRMaj rect1 = rectifyAlg.getUndistToRectPixels1();
        DMatrixRMaj rect2 = rectifyAlg.getUndistToRectPixels2();
        // New calibration matrix,
        DMatrixRMaj rectK = rectifyAlg.getCalibrationMatrix();

        // Adjust the rectification to make the view area more useful
        RectifyImageOps.allInsideLeft(param.left, rect1, rect2, rectK, null);

        // Save rectification for later use
        dparam.disparityMin = disparityMin;
        dparam.disparityRange = disparityRange;
        dparam.baseline = param.getBaseline();
        PerspectiveOps.matrixToPinhole(rectK, rectLeft.width, rectRight.height, dparam.pinhole);
        dparam.rotateToRectified.setTo(rectifyAlg.getRectifiedRotation());

        // undistorted and rectify images
        var rect1_F32 = new FMatrixRMaj(3, 3);
        var rect2_F32 = new FMatrixRMaj(3, 3);
        ConvertMatrixData.convert(rect1, rect1_F32);
        ConvertMatrixData.convert(rect2, rect2_F32);

        ImageDistort<GrayF32, GrayF32> imageDistortLeft =
                RectifyDistortImageOps.rectifyImage(param.getLeft(), rect1_F32, BorderType.SKIP, origLeft.getImageType());
        ImageDistort<GrayF32, GrayF32> imageDistortRight =
                RectifyDistortImageOps.rectifyImage(param.getRight(), rect2_F32, BorderType.SKIP, origRight.getImageType());

        imageDistortLeft.apply(origLeft, rectLeft);
        imageDistortRight.apply(origRight, rectRight);

        return rectifyAlg;
    }

    private static void printHelpExit(CmdLineParser parser) {
        parser.getProperties().withUsageWidth(120);
        parser.printUsage(System.out);
        System.out.println("Computes stereo disparity for one or more stereo image pairs.");
        System.out.println();
        System.out.println("Image paths can point to a file, directory, glob, or regex. Examples:");
        System.out.println("   Glob example: 'glob:data/**/left*.jpg'");
        System.out.println("   Regex example: 'regex:data/\\w+/left\\d+.jpg'");
        System.out.println();
        System.out.println("Calibration needs to be a BoofCV formatted yaml file for stereo cameras");
    }

    public static void main(String[] args) {
        var generator = new TrinocularDisparityApp();
        var parser = new CmdLineParser(generator);

        if (args.length == 0) {
            printHelpExit(parser);
            return;
        }

        try {
            parser.parseArgument(args);
            try {
                generator.process();
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println();
                System.out.println("Failed! See exception above");
            }
        } catch (CmdLineException e) {
            // handling of wrong arguments
            System.err.println(e.getMessage());
            printHelpExit(parser);
        }
    }
}
