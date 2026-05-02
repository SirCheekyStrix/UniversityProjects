#include <iostream>
#include <vector>
#include <stack>
#include <fstream>
#include <sstream>

struct Point {
    int x, y;
};

//base point finder
Point findPivot(const std::vector<Point>& points) {
    Point pivot = points[0];
    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].y < pivot.y || (points[i].y == pivot.y && points[i].x < pivot.x)) {
            pivot = points[i];
        }
    }
    return pivot;
}

//function to find orientation
int det(const Point& p, const Point& q, const Point& r) {
    return (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
}

//point sorting by angle
void angleSort(std::vector<Point>& points, const Point& pivot) {
    for (size_t i = 1; i < points.size(); ++i) {
        for (size_t j = 1; j < points.size() - i + 1; ++j) {
            int orient = det(pivot, points[j - 1], points[j]);
            if (orient > 0 || (orient == 0 && (pivot.x - points[j - 1].x) * (pivot.x - points[j - 1].x) + (pivot.y - points[j - 1].y) * (pivot.y - points[j - 1].y) >
                                              (pivot.x - points[j].x) * (pivot.x - points[j].x) + (pivot.y - points[j].y) * (pivot.y - points[j].y))) {
                std::swap(points[j - 1], points[j]);
            }
        }
    }
}

//Graham's algorithm for convex hull
std::vector<Point> convexHull(std::vector<Point>& points) {
    int n = points.size();
    if (n < 3) throw std::invalid_argument("Potrzebne są co najmniej 3 punkty");

    //finding lowest point
    Point pivot = findPivot(points);
    std::swap(points[0], pivot);

    //sorting points by angle
    angleSort(points, points[0]);


    std::stack<Point> hull;
    hull.push(points[0]);
    hull.push(points[1]);

    for (int i = 2; i < n; ++i) {
        while (hull.size() >= 2) {
            Point second = hull.top(); hull.pop();
            Point first = hull.top();
            if (det(first, second, points[i]) < 0) {
                hull.push(second);
                break;
            }
        }
        hull.push(points[i]);
    }

    std::vector<Point> result;
    while (!hull.empty()) {
        result.push_back(hull.top());
        hull.pop();
    }
    return result;
}

std::vector<Point> readFromCSV(const std::string& filename) { //remove in final version
    std::vector<Point> points;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::string line;
    std::getline(file, line); //headline skip
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string x_str, y_str;
        if (std::getline(ss, x_str, ',') && std::getline(ss, y_str, ',')) {
            Point point = {std::stoi(x_str), std::stoi(y_str)};
            points.push_back(point);
        }
    }
    file.close();
    return points;
}

std::vector<std::vector<Point>> classifyPointsIntoQuadrants(const std::vector<Point>& points, int max_x, int max_y) {
    std::vector<Point> quadrant1, quadrant2, quadrant3, quadrant4;

    
    //diving points into quadrants
    for (size_t i = 0; i < points.size(); ++i) {
        const Point& point = points[i];
        //quadrant 1 = top-left
        if (point.x <= max_x / 2 && point.y > max_y / 2) {
            quadrant1.push_back(point);
        }
        //quadrant 2 = top-right
        else if (point.x > max_x / 2 && point.y > max_y / 2) {
            quadrant2.push_back(point);
        }
        //quadrant 3 = bottom-left
        else if (point.x <= max_x / 2 && point.y <= max_y / 2) {
            quadrant3.push_back(point);
        }
        //quadrant 4 = bottom-right
        else if (point.x > max_x / 2 && point.y <= max_y / 2) {
            quadrant4.push_back(point);
        }
    }

    return {quadrant1, quadrant2, quadrant3, quadrant4};
}


int main() {
    std::string filename = "Points.csv";

    try {
        std::vector<Point> points = readFromCSV(filename); //reading points

        //max x and y are max coords where the fields are located (to change in final version???)
        int max_x = 50;
        int max_y = 100;
        auto quadrants = classifyPointsIntoQuadrants(points, max_x, max_y);

        //for each quadrant calculating convex hull
        for (size_t i = 0; i < quadrants.size(); ++i) {
            if (quadrants[i].size() >= 3) {
                std::cout << "Convex hull for Quadrant " << i + 1 << ":\n";
                auto hull = convexHull(quadrants[i]);
                for (size_t j = 0; j < hull.size(); ++j) {
                    std::cout << "(" << hull[j].x << ", " << hull[j].y << ")\n";
                }
                std::cout << std::endl;
            } else {
                std::cout << "Quadrant " << i + 1 << " does not have enough points for a convex hull.\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}