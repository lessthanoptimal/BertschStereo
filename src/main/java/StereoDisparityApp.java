import boofcv.abst.disparity.ConfigSpeckleFilter;
import boofcv.abst.disparity.DisparitySmoother;
import boofcv.abst.disparity.StereoDisparity;
import boofcv.alg.cloud.PointCloudReader;
import boofcv.alg.cloud.PointCloudWriter;
import boofcv.alg.distort.ImageDistort;
import boofcv.alg.filter.misc.AverageDownSampleOps;
import boofcv.alg.geo.PerspectiveOps;
import boofcv.alg.geo.RectifyDistortImageOps;
import boofcv.alg.geo.RectifyImageOps;
import boofcv.alg.geo.rectify.RectifyCalibrated;
import boofcv.alg.mvs.DisparityParameters;
import boofcv.alg.mvs.MultiViewStereoOps;
import boofcv.factory.disparity.ConfigDisparityBMBest5;
import boofcv.factory.disparity.DisparityError;
import boofcv.factory.disparity.FactoryStereoDisparity;
import boofcv.gui.image.ShowImages;
import boofcv.gui.image.VisualizeImageData;
import boofcv.io.UtilIO;
import boofcv.io.calibration.CalibrationIO;
import boofcv.io.image.UtilImageIO;
import boofcv.io.points.PointCloudIO;
import boofcv.misc.BoofMiscOps;
import boofcv.struct.border.BorderType;
import boofcv.struct.calib.StereoParameters;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import boofcv.visualize.PointCloudViewer;
import boofcv.visualize.VisualizeData;
import georegression.struct.se.Se3_F64;
import org.apache.commons.io.FilenameUtils;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.FMatrixRMaj;
import org.ejml.ops.ConvertMatrixData;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;

/**
 * @author Peter Abeles
 */
public class StereoDisparityApp {
    @Option(name = "-l", aliases = {"--Left"}, usage = "Location of stereo left images")
    public String pathLeft;

    @Option(name = "-r", aliases = {"--Right"}, usage = "Location of stereo right images")
    public String pathRight;

    @Option(name = "-c", aliases = {"--Calibration"}, usage = "Location of stereo calibration")
    public String pathCalibration;

    @Option(name = "-o", aliases = {"--Output"}, usage = "Path to output directory")
    public String pathOutput = "stereo_output";

    @Option(name = "--MaxWidth", usage = "If an image has a width larger than this it will be scaled")
    public int maxWidth = 640;

    @Option(name = "--Display", usage = "Displays the 3D cloud in a window")
    public boolean display = false;

    public int regionSize = 5;
    public int disparityMin = 0;
    public int disparityRange = 255;

    RectifyCalibrated rectifyAlg = RectifyImageOps.createCalibrated();
    DisparityParameters dparam = new DisparityParameters();

    public void process() {
        var dirOutput = new File(pathOutput);
        if (!dirOutput.exists())
            BoofMiscOps.checkTrue(dirOutput.mkdirs());

        List<String> inputsLeft = UtilIO.listSmart(pathLeft, true, (f) -> true);
        List<String> inputsRight = UtilIO.listSmart(pathRight, true, (f) -> true);

        if (inputsLeft.size() != inputsRight.size())
            throw new RuntimeException(String.format("Number of left=%d and right=%d images do not match.",
                    inputsLeft.size(), inputsRight.size()));

        if (inputsLeft.isEmpty()) {
            System.err.println("No input images found. Check the path?");
            System.exit(2);
        }

        StereoParameters calibration = CalibrationIO.load(pathCalibration);
        StereoParameters scaledCalibration = new StereoParameters(calibration);

        // Storage for scaled images
        var scaledLeft = new GrayU8(1, 1);
        var scaledRight = new GrayU8(1, 1);
        // storage for rectified images
        var rectLeft = new GrayU8(1, 1);
        var rectRight = new GrayU8(1, 1);
        var disparity = new GrayF32(1, 1);

        for (int imageIdx = 0; imageIdx < inputsLeft.size(); imageIdx++) {
            GrayU8 left = UtilImageIO.loadImage(inputsLeft.get(imageIdx), GrayU8.class);
            GrayU8 right = UtilImageIO.loadImage(inputsRight.get(imageIdx), GrayU8.class);

            if (left == null || right == null) {
                System.err.println("Failed to load image " + inputsLeft.get(imageIdx));
                continue;
            }
            String name = FilenameUtils.getBaseName(inputsLeft.get(imageIdx));

            scaledCalibration.setTo(calibration);
            if (left.width > maxWidth) {
                double scale = maxWidth / (double) left.width;
                scaledLeft.reshape(maxWidth, left.height * maxWidth / left.width);
                AverageDownSampleOps.down(left, scaledLeft);
                scaledRight.reshape(maxWidth, right.height * maxWidth / right.width);
                AverageDownSampleOps.down(right, scaledRight);
                PerspectiveOps.scaleIntrinsic(scaledCalibration.left, scale);
                PerspectiveOps.scaleIntrinsic(scaledCalibration.right, scale);
            } else {
                scaledLeft.setTo(left);
                scaledRight.setTo(right);
            }

            System.out.printf("%-30s %dx%d -> %dx%d\n", name, left.width, left.height, scaledLeft.width, scaledLeft.height);

            rectify(scaledLeft, scaledRight, scaledCalibration, rectLeft, rectRight);

            denseDisparitySubpixel(rectLeft, rectRight, regionSize, disparityMin, disparityRange,
                    disparity);

            // Here's what we came here for. Time to remove the speckle
            var configSpeckle = new ConfigSpeckleFilter();
            configSpeckle.similarTol = 1.0f; // Two pixels are connected if their disparity is this similar
            configSpeckle.maximumArea.setFixed(200); // probably the most important parameter, speckle size
            DisparitySmoother<GrayU8, GrayF32> smoother =
                    FactoryStereoDisparity.removeSpeckle(configSpeckle, GrayF32.class);

            smoother.process(rectLeft, disparity, disparityRange);

            // Colorize disparity in a way that's easy for a person to understand
            BufferedImage visualizedDisparity = VisualizeImageData.disparity(disparity, null, disparityRange, 0);

            UtilImageIO.saveImage(visualizedDisparity, new File(dirOutput, name + "_visualized.png").getPath());

            // Compute point cloud
            var cloud = new PointCloudWriter.CloudArraysF32();
            MultiViewStereoOps.disparityToCloud(disparity, dparam, null,
                    (pixX, pixY, x, y, z) -> {
                        int v = rectLeft.get(pixX, pixY);
                        cloud.add(x, y, z, v << 16 | v << 8 | v);
                    });

            // Save cloud as PLY
            try (var out = new FileOutputStream(new File(dirOutput, "cloud.ply"))) {
                PointCloudIO.save3D(PointCloudIO.Format.PLY, PointCloudReader.wrap((idx, p) -> {
                    p.rgb = cloud.cloudRgb.get(idx);
                    p.x = cloud.cloudXyz.data[idx * 3];
                    p.y = cloud.cloudXyz.data[idx * 3 + 1];
                    p.z = cloud.cloudXyz.data[idx * 3 + 2];
                }, cloud.cloudRgb.size), false, out);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            if (display) {
                PointCloudViewer guiPointCloud = VisualizeData.createPointCloudViewer();
                guiPointCloud.setCameraHFov(PerspectiveOps.computeHFov(dparam.pinhole));
                guiPointCloud.addCloud(cloud::getPoint, cloud.cloudRgb::get, cloud.cloudRgb.size);
                JComponent component = guiPointCloud.getComponent();
                component.setPreferredSize(new Dimension(dparam.pinhole.width, dparam.pinhole.height));
                ShowImages.showWindow(component, "Point Cloud", true);
            }
        }
    }

    public static void denseDisparitySubpixel(GrayU8 rectLeft, GrayU8 rectRight,
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
//        config.validateRtoL = 1;
        config.texture = 0.2;
        StereoDisparity<GrayU8, GrayF32> disparityAlg =
                FactoryStereoDisparity.blockMatchBest5(config, GrayU8.class, GrayF32.class);

        // process and return the results
        disparityAlg.process(rectLeft, rectRight);

        disparity.setTo(disparityAlg.getDisparity());
    }

    /**
     * Rectified the input images using known calibration.
     */
    public RectifyCalibrated rectify(GrayU8 origLeft, GrayU8 origRight,
                                     StereoParameters param,
                                     GrayU8 rectLeft, GrayU8 rectRight) {
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

        ImageDistort<GrayU8, GrayU8> imageDistortLeft =
                RectifyDistortImageOps.rectifyImage(param.getLeft(), rect1_F32, BorderType.SKIP, origLeft.getImageType());
        ImageDistort<GrayU8, GrayU8> imageDistortRight =
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
        var generator = new StereoDisparityApp();
        var parser = new CmdLineParser(generator);

        if (args.length == 0) {
            printHelpExit(parser);
        }

        try {
            parser.parseArgument(args);
            if (generator.pathLeft == null) {
                System.err.println("You must specify path to left camera images");
                System.exit(1);
            }
            if (generator.pathRight == null) {
                System.err.println("You must specify path to right camera images");
                System.exit(1);
            }
            if (generator.pathCalibration == null) {
                System.err.println("You must specify calibration file");
                System.exit(1);
            }
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
