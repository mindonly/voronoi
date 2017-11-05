#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <random>
#include <cmath>
#include <map>

#include "bitmap_image.hpp"     // C++ Bitmap lib: http://partow.net/programming/bitmap/index.html

#define MAX_DIST 99999;

using std::cout;
using std::cerr;


typedef std::pair<int, int> pixel;

std::vector<pixel> generateSeeds(int nseeds, int wd, int ht);
double euclidean(pixel &p1, pixel &p2);
int    manhattan(pixel &p1, pixel &p2);
pixel closest_euc_seed(std::vector<pixel> &seedvec, pixel &p);
pixel closest_man_seed(std::vector<pixel> &seedvec, pixel &p);
void printPixel(pixel &px);
void printSeeds(std::vector<pixel> &seedvec);
void color_bitmap(int img_wd, int img_ht,
                  const std::vector<pixel> &seeds, 
                  const std::map<pixel, rgb_t> &cmap,
                  bitmap_image &target_bmp);

/*
 * Timer class from https://gist.github.com/gongzhitaao/7062087
 * Timer() constructs the timer
 * .reset() resets the timer
 * .elapsed() returns elapsed seconds in double since last reset
 */
class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const { 
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count(); }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};

/*
 * Euclidean distance functor for std::transform and thrust::transform
 */
struct ClosestEuclideanSeed {
    pixel operator()(std::vector<pixel> &seedvec, pixel &p) const 
    {
        pixel closest = std::make_pair(-1, -1);
        double closest_dist = MAX_DIST;
    
        for (auto &seed_px : seedvec) { 
            double cur_dist = euclidean(seed_px, p);
            if (cur_dist < closest_dist) {
                closest_dist = cur_dist;
                closest = seed_px;
            }
        }
    
        return closest;
    }
};

/*
 * Manhattan distance functor for std::transform and thrust::transform
 */
struct ClosestManhattanSeed {
    pixel operator()(std::vector<pixel> &seedvec, pixel &p) const
    {
        pixel closest = std::make_pair(-1, -1);
        double closest_dist = MAX_DIST;
    
        for (auto &seed_px : seedvec) { 
            double cur_dist = manhattan(seed_px, p);
            if (cur_dist < closest_dist) {
                closest_dist = cur_dist;
                closest = seed_px;
            }
        }
    
        return closest;
    }
};

    // main program
int main(int argc, char* argv[])
{
    if (argc != 4) {
        cerr << "incorrect number of arguments! \n\tusage: ./voronoi <seeds> <width> <height>\n";
        exit(-1);
    }

        // process command-line arguments
    int num_seeds  = atoi(argv[1]);
    int IMG_WIDTH  = atoi(argv[2]);
    int IMG_HEIGHT = atoi(argv[3]);

        // instantiate the timer
    Timer tmr;

        // create Euclidean and Manhattan bitmaps
    bitmap_image voronoi_euclidean(IMG_WIDTH, IMG_HEIGHT);
    bitmap_image voronoi_manhattan(IMG_WIDTH, IMG_HEIGHT);

        // generate and sort the image seeds
    std::vector<pixel> seeds = generateSeeds(num_seeds, IMG_WIDTH, IMG_HEIGHT);
    std::sort(seeds.begin(), seeds.end());

        // set up matrix of seeds (vector of seed vectors) for std::transform and thrust::transform
    std::vector <pixel> *pv;
    pv = &seeds;
    std::vector<std::vector <pixel> > seeds_matrix;
    seeds_matrix.push_back(seeds);
    for (int i=1; i < IMG_WIDTH * IMG_HEIGHT; i++) {
        seeds_matrix.push_back(*pv);
    }

        // set up vector of image pixels
    std::vector <pixel> img_pixels;
    for (int y=0; y < IMG_HEIGHT; y++) {
        for (int x=0; x < IMG_WIDTH; x++) {
            img_pixels.push_back(std::make_pair(x, y));
        }
    }

        // set up color map
    std::map<pixel, rgb_t> color_map;
    for (auto &p : seeds) {
        rgb_t color = make_colour(rand() % 255, rand() % 255, rand() % 255);
        voronoi_euclidean.set_pixel(p.first, p.second, color);
        voronoi_manhattan.set_pixel(p.first, p.second, color);
        color_map.emplace(p, color);
    }

        // create target vectors for closest seed results
    std::vector<pixel> closest_euclidean_seeds(IMG_WIDTH * IMG_HEIGHT);
    std::vector<pixel> closest_manhattan_seeds(IMG_WIDTH * IMG_HEIGHT);

        // record the setup time
    double setup_time = tmr.elapsed();
    
        // reset the timer
    tmr.reset();

        // apply distance functors with transform() to seeds matrix and pixel vector
    std::transform(seeds_matrix.begin(), seeds_matrix.end(), img_pixels.begin(), 
                   closest_euclidean_seeds.begin(), ClosestEuclideanSeed());
    std::transform(seeds_matrix.begin(), seeds_matrix.end(), img_pixels.begin(), 
                   closest_manhattan_seeds.begin(), ClosestManhattanSeed());

        // record the transformation time
    double trans_time = tmr.elapsed();

        // reset the timer
    tmr.reset();

        // color the target bitmaps (Euclidean & Manhattan)
    color_bitmap(IMG_WIDTH, IMG_HEIGHT, closest_euclidean_seeds, color_map, voronoi_euclidean);
    color_bitmap(IMG_WIDTH, IMG_HEIGHT, closest_manhattan_seeds, color_map, voronoi_manhattan);

        // write the Euclidean and Manhattan bitmaps
    voronoi_euclidean.save_image("voronoi_euclidean.bmp");
    voronoi_manhattan.save_image("voronoi_manhattan.bmp");

        // record the image coloring and writing time
    double color_write_time = tmr.elapsed();

        // timings
    cout << "      SEEDS: " << num_seeds << '\n';
    cout << " DIMENSIONS: " << IMG_WIDTH << 'x' << IMG_HEIGHT <<'\n';
    cout << "        setup time: " << setup_time       * 1000 << " ms \n";
    cout << "    transform time: " << trans_time       * 1000 << " ms \n";
    cout << "color & write time: " << color_write_time * 1000 << " ms \n";

    return 0;
}

    // generate Voronoi seeds
std::vector<pixel> generateSeeds(int nseeds, int wd, int ht)
{
    std::random_device rd;
    std::default_random_engine rng(rd());

    std::uniform_int_distribution<> x_dim(0, wd - 1);
    std::uniform_int_distribution<> y_dim(0, ht - 1);

    std::vector<pixel> sv;
    for (int i=0; i < nseeds; i++) {
        int x_coord = x_dim(rng);
        int y_coord = y_dim(rng);
   
        pixel p = std::make_pair(x_coord, y_coord);
        sv.push_back(p);
    }

    return sv;
}

    // color the target bitmap
void color_bitmap(int img_wd, int img_ht,
                  const std::vector<pixel> &seeds, 
                  const std::map<pixel, rgb_t> &cmap,
                  bitmap_image &target_bmp) 
{
    for (int y=0; y < img_ht; y++) {
        for (int x=0; x < img_wd; x++) {
            const pixel *pv = seeds.data();    // get the seeds vector raw data
            pixel p = pv[y * img_wd + x];
            auto search = cmap.find(p);
            rgb_t color = search->second;
            target_bmp.set_pixel(x, y, color);
        }
    }
}

	// compute Euclidean distance between pixels
double euclidean(pixel &p1, pixel &p2)
{
	int x1 = p1.first;
	int y1 = p1.second;
	int x2 = p2.first;
	int y2 = p2.second;
	
	return sqrt( pow(x2 - x1, 2) + pow(y2 - y1, 2) );
}

	// compute Manhattan distance between pixels
int manhattan(pixel &p1, pixel &p2)
{
	int x1 = p1.first;
	int y1 = p1.second;
	int x2 = p2.first;
	int y2 = p2.second;
	
	return abs(x2 - x1) + abs(y2 - y1);
}

    // print Voronoi seeds
void printSeeds(std::vector<pixel> &sv)
{
    for (auto &p : sv)
        cout << '[' << p.first << ", " << p.second << "]\n";

    cout << '\n';
}
    
        // print a pixel
void printPixel(pixel &px)
{
    cout << '[' << px.first << ", " << px.second << "]";
}
