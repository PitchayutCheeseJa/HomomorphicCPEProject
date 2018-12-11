import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import java.util.*;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class Cheese {

    public static void main(String[] argd) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Scanner scan = new Scanner(System.in);
        Mat image = Imgcodecs.imread("C:\\Users\\pc\\Desktop\\Spine Segmentation\\test_homomorphic.jpg", Imgcodecs.IMREAD_GRAYSCALE);
        if (image.empty()) {
            System.out.println("Error opening image");
            System.exit(-1);
        }
        displayImage(matToBufferedImage(image), "Input Image");
        Imgcodecs.imwrite("C:\\Users\\pc\\Desktop\\Spine Segmentation\\test1_grey.jpg", image);
        Mat padded = new Mat();
        int m = Core.getOptimalDFTSize(image.rows());
        int n = Core.getOptimalDFTSize(image.cols());
        Core.copyMakeBorder(image, padded, 0, m - image.rows(), 0, n - image.cols(), Core.BORDER_CONSTANT, Scalar.all(0));
        List<Mat> planes = new ArrayList<Mat>();
        padded.convertTo(padded, CvType.CV_32F);
        //===================================================================================
        //=====================================Log===========================================
        //===================================================================================
        
        padded = change_To_log(padded);  //log
        
        //===================================================================================
        //=====================================DFT===========================================
        //===================================================================================
        planes.add(padded);
        planes.add(Mat.zeros(padded.size(), CvType.CV_32F));
        Mat complexI = new Mat(padded.size(), CvType.CV_32F);
        Core.merge(planes, complexI);
        Core.dft(complexI, complexI);
        Core.split(complexI, planes);
        Mat Real = new Mat();
        planes.get(0).copyTo(Real);
        Mat Imagine = new Mat();
        planes.get(1).copyTo(Imagine);
        //===================================================================================
        //==================================Quadrant=========================================
        //===================================================================================
        quadrant(Real);
        quadrant(Imagine);
        //===================================================================================
        //==================================Add Pixel========================================
        //===================================================================================
        int rowsR = Real.rows();
        int colsR = Real.cols();
        int chR = Real.channels();
        int midrow = rowsR / 2 + 1;
        int midcol = colsR / 2 + 1;
        int ksize;
        System.out.print("ksize = ");
        ksize = scan.nextInt();
        int addksize = ksize / 2;
        for (int i = 0; i < rowsR; i++) {
            for (int j = 0; j < colsR; j++) {
                double[] data1 = Real.get(i, j);
                double[] data2 = Imagine.get(i, j);
                for (int k = 0; k < chR; k++)
                {
                    if ((i >= midrow - addksize && i <= midrow + addksize) && (j >= midcol - addksize && j <= midcol + addksize)) {
                        
                        data1[k] = 0.0;
                        data2[k] = 0.0;
                        //i < midrow - addksize || i > midrow + addksize || j < midcol - addksize || j > midcol + addksize
                        //(i >= midrow - addksize && i <= midrow + addksize) && (j >= midcol - addksize && j <= midcol + addksize)
                    }
                }
                Real.put(i, j, data1);
                Imagine.put(i, j, data2);
            }
        }
        //===================================================================================
        //==================================Quadrant=========================================
        //===================================================================================
        quadrant(Real);
        quadrant(Imagine);
        //===================================================================================
        //====================================IDFT==========================================
        //===================================================================================
        List<Mat> Real_and_Imagine = new ArrayList<Mat>();
        Real_and_Imagine.add(Real);
        Real_and_Imagine.add(Imagine);
        Mat Complex2 = new Mat();
        Core.merge(Real_and_Imagine, Complex2);
        Core.idft(Complex2, Complex2);
        Core.split(Complex2, Real_and_Imagine);
        Mat final_image = new Mat();
        Real_and_Imagine.get(0).copyTo(final_image);
        //===================================================================================
        //=====================================expo==========================================
        //===================================================================================
        Core.normalize(final_image, final_image, 0, 1, Core.NORM_MINMAX, CvType.CV_32F);
        Core.exp(final_image, final_image);
        //===================================================================================
        //=====================================Show==========================================
        //===================================================================================
        show_image(final_image,"Result");
        Imgcodecs.imwrite("C:\\Users\\pc\\Desktop\\Spine Segmentation\\test1_result.jpg", final_image);  
    }
    
    public static void show_image(Mat x,String title){
        Core.normalize(x, x, 0, 255, Core.NORM_MINMAX, CvType.CV_32F);
        x.convertTo(x, CvType.CV_8UC1);
        displayImage(matToBufferedImage(x), title);
    }
    public static void quadrant(Mat x){
         x = x.submat(new Rect(0, 0, x.cols() & -2, x.rows() & -2));
        int cx = x.cols() / 2;
        int cy = x.rows() / 2;

        Mat q0 = new Mat(x, new Rect(0, 0, cx, cy));
        Mat q1 = new Mat(x, new Rect(cx, 0, cx, cy));
        Mat q2 = new Mat(x, new Rect(0, cy, cx, cy));
        Mat q3 = new Mat(x, new Rect(cx, cy, cx, cy));

        Mat tmp = new Mat();
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }
    public static Mat change_To_log(Mat x) {
        int rows = x.rows(); //Calculates number of rows
        int cols = x.cols(); //Calculates number of columns
        int ch = x.channels(); //Calculates number of channels (Grayscale: 1, RGB: 3, etc.)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double[] data = x.get(i, j); //Stores element in an array
                for (int k = 0; k < ch; k++) //Runs for the available number of channels
                {

                    data[k] += 1;
                    //data[k] += 1.0; //Pixel modification done here

                }
                x.put(i, j, data);
            }
        }

        Core.log(x, x);  //log
        return x;
    }
    public static void print_Matric(Mat x) {
        int rows = x.rows(); //Calculates number of rows
        int cols = x.cols(); //Calculates number of columns
        int ch = x.channels(); //Calculates number of channels (Grayscale: 1, RGB: 3, etc.)
        System.out.println("Rows = " + rows + " cols = " + cols + " ch = " + ch);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double[] data = x.get(i, j); //Stores element in an array
                for (int k = 0; k < ch; k++) //Runs for the available number of channels
                {
                    System.out.print(data[k] + " ");
                }
                x.put(i, j, data);
            }
            System.out.println("");
        }
        System.out.println("");
        System.out.println("");
    }
    public static BufferedImage matToBufferedImage(Mat bgr) {
        int width = bgr.width();
        int height = bgr.height();
        BufferedImage image;
        WritableRaster raster;

        if (bgr.channels() == 1) {
            image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            raster = image.getRaster();

            byte[] px = new byte[1];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    bgr.get(y, x, px);
                    raster.setSample(x, y, 0, px[0]);
                }
            }
        } else {
            image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            raster = image.getRaster();

            byte[] px = new byte[3];
            int[] rgb = new int[3];

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    bgr.get(y, x, px);
                    rgb[0] = px[2];
                    rgb[1] = px[1];
                    rgb[2] = px[0];
                    raster.setPixel(x, y, rgb);
                }
            }
        }

        return image;
    }
    public static void displayImage(Image img2, String title) {
        //BufferedImage img=ImageIO.read(new File("/HelloOpenCV/lena.png"));
        ImageIcon icon = new ImageIcon(img2);
        JFrame frame = new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(img2.getWidth(null) + 200, img2.getHeight(null) + 200);
        JLabel lbl = new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setTitle(title);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    }
}
