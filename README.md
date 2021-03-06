# CoffeeGrindSieve

CoffeeGrindSieve is designed to help the barista optimize the coffee brewing process by measuring the size distribution of the coffee grind particles. The coffee brewing process is influenced by several factors, including brewing time, water pressure and temperature and the coffee grind size. The latter is mutually adjusted with the others to find the sweet spot of perfect taste for a given coffee bean roast and brewing technology (see e.g. https://www.homegrounds.co/coffee-grind-chart or https://ineedcoffee.com/coffee-grind-chart). 

The input data is an image of suitably prepared coffee grind taken using a low magnification microscope. MM operations including erosion, dilation, opening, closing, tophat, geodesic distance and waterfall transform are used to identify and isolate grind particles such that their size in terms of pixels can be measured.

