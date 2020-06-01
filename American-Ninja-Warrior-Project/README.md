
Now that you are here, you are about to witness a succinct and effective cleaning of the data regarding the history of obstacles of the ten seasons of American Ninja Warrior. There will also be some cool analysis to top it off.

# It's really cool, I know.

---








### I remember watching the original Ninja Warrior that aired in Japan when I was younger. Both the original and the American version are great showcases of the capacity of human athleticism and also great showcases of my current non-so-athletic ability. But who knows, I'm only a junior in college (as of 4/12/2019), anything can happen!

I downloaded the file "American Ninja Warrior Obstacle History.xlsx" from the Data.World (https://data.world/ninja/anw-obstacle-history). If you choose to follow along, the file is right there for the taking.


Because I use Google Colab instead of Jupyter notebook, I could just save the relevant files into my Google Drive and then mount them onto Colab, allowing me to access them without needing to save to my local machine. 

The next few lines are specific to my workflow and are not necessary for everyone to emulate.


```
from google.colab import files
```


```
from google.colab import drive
```


```
drive.mount('/content/gdrive')
```

    Go to this URL in a browser: https://accounts.google.com/o/oauth2/auth?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&redirect_uri=urn%3Aietf%3Awg%3Aoauth%3A2.0%3Aoob&scope=email%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdocs.test%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive.photos.readonly%20https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fpeopleapi.readonly&response_type=code
    
    Enter your authorization code:
    ··········
    Mounted at /content/gdrive



```
import numpy as np
import pandas as pd
```

It is necessary to import these libraries, as they are pretty much ubiqituous to all data science workflows.


```
filePath = '/content/gdrive/My Drive/Google Colaboratory/Files for Google Colab/American Ninja Warrior Project'
amwDat = pd.read_csv(filePath +'/amw.csv', error_bad_lines=False)
```

    b'Skipping line 891: expected 5 fields, saw 6\nSkipping line 892: expected 5 fields, saw 6\nSkipping line 893: expected 5 fields, saw 6\nSkipping line 894: expected 5 fields, saw 6\nSkipping line 895: expected 5 fields, saw 6\nSkipping line 896: expected 5 fields, saw 6\nSkipping line 897: expected 5 fields, saw 6\nSkipping line 898: expected 5 fields, saw 6\nSkipping line 899: expected 5 fields, saw 6\nSkipping line 900: expected 5 fields, saw 6\nSkipping line 901: expected 5 fields, saw 6\nSkipping line 902: expected 5 fields, saw 6\nSkipping line 903: expected 5 fields, saw 6\nSkipping line 904: expected 5 fields, saw 6\nSkipping line 905: expected 5 fields, saw 6\n'



```
amwDat.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Location</th>
      <th>Round/Stage</th>
      <th>Obstacle Name</th>
      <th>Obstacle Order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Quintuple Steps</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Rope Swing</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Rolling Barrel</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Jumping Spider</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Pipe Slider</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



With the head() function, I get a general idea of what the data is like. This dataset in particular is quite small and not representative of the super crazy huge sets that most businesses work with. 

### I can see that there are five variables (or features): Season, Location, Round/Stage, Obstacle Name, Obstacle Order.

## Let's dig a little deeper.


```
amwDat.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Obstacle Order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>889.000000</td>
      <td>889.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.422947</td>
      <td>4.577053</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.573901</td>
      <td>2.583509</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>7.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9.000000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>10.000000</td>
      <td>10.000000</td>
    </tr>
  </tbody>
</table>
</div>




```
amwDat['Season'] = amwDat['Season'].astype('int')
amwDat['Obstacle Order'] = amwDat['Obstacle Order'].astype('int')
```

Here, we can convert the columns Season and Obstacle into integer vectors


```
amwDat.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 889 entries, 0 to 888
    Data columns (total 5 columns):
    Season            889 non-null int64
    Location          889 non-null object
    Round/Stage       889 non-null object
    Obstacle Name     889 non-null object
    Obstacle Order    889 non-null int64
    dtypes: int64(2), object(3)
    memory usage: 34.8+ KB


What we have here is a dataframe with five columns, each with 889 entries. Two of the columns, season and obstacle order, are integer vectors, and three of the columns, location, round/stage, and obstacle name, are object vectors.

## Let's take a look at the unique values in each column


```
colNames = list(amwDat)
for col in colNames:
  print(col)
  print(amwDat[col].unique())
  print('\n')
```

    Season
    [ 1  2  3  4  5  6  7  8  9 10]
    
    
    Location
    ['Venice' 'Sasuke 23 (Japan)' 'Sasuke 26 (Japan)' 'Sasuke 27 (Japan)'
     'Southwest' 'Midwest' 'Northeast' 'Northwest' 'Mid-South' 'Southeast'
     'Las Vegas' 'Baltimore' 'Miami' 'Denver' 'Dallas' 'St. Louis'
     'Pittsburgh' 'Orlando' 'Kansas City' 'Houston' 'San Pedro (Military)'
     'Los Angeles' 'Atlanta' 'Indianapolis' 'Oklahoma City' 'Philadelphia'
     'San Antonio' 'Daytona Beach' 'Cleveland' 'Minneapolis']
    
    
    Round/Stage
    ['Qualifying' 'Semi-Finals' 'National Finals - Stage 1'
     'National Finals - Stage 2' 'National Finals - Stage 3'
     'National Finals - Stage 4' 'Qualifying (Regional/City)'
     'Finals (Regional/City)']
    
    
    Obstacle Name
    ['Quintuple Steps' 'Rope Swing' 'Rolling Barrel' 'Jumping Spider'
     'Pipe Slider' 'Warped Wall' 'Tarzan Swing' 'Jumping Bars' 'Cargo Climb'
     'Twelve Timbers' 'Curtain Slider' 'Log Grip' 'Half-Pipe Attack'
     'Slider Jump' 'Tarzan Rope' 'Rope Ladder' 'Downhill Jump' 'Salmon Ladder'
     'Stick Slider' 'Unstable Bridge' 'Metal Spin' 'Wall Lift' 'Arm Rings'
     'Descending Lamp Grasper' 'Devil Steps' 'Shin Cliffhanger' 'Hang Climb'
     'Spider Flip' 'Gliding Ring' 'Heavenly Ladder' 'G-Rope' 'Quad Steps'
     'Bridge of Blades' 'Circle Slider' 'Step Slider' 'Hazard Swing'
     'Rolling Escargot' 'Giant Swing' 'Slider Drop' 'Double Salmon Ladder'
     'Balance Tank' 'Roulette Cylinder' 'Doorknob Grasper' 'Cycle Road'
     'Ultimate Cliffhanger' 'Swing Circle' 'Bungee Rope Climb' 'Flying Bar'
     'Rope Climb' 'Jump Hang' 'Spinning Bridge' 'Arm Bike' 'Jumping Rings'
     'Chain Seesaw' 'Bar Glider' 'Spinning Log' 'Lamp Grasper' 'Bungee Bridge'
     'Pipe Slider / Devil Steps' 'Rope Junction' 'Rolling Log'
     'Floating Boards' 'Frame Slider' 'Domino Hill' 'Floating Chains'
     'Flying Nunchuks / Trapeze Swing' 'Rope Maze' 'Cliffhanger'
     'Spider Climb' 'Prism Tilt' 'Swing Jump' 'Circle Cross' 'Rumbling Dice'
     'Body Prop' 'Utility Pole Slider' 'Balance Bridge' 'Monkey Pegs'
     'Ledge Jump' 'Rolling Steel' 'Rotating Bridge' 'Jump Hang Kai'
     'Grip Hang' 'Floating Stairs' 'Pole Grasper' 'Timbers' 'Giant Ring'
     'Rope Glider' 'Hang Slider' 'Spinning Wheel' 'Slack Ladder'
     'Jumping Bars into Cargo Net' 'Cannonball Alley' 'Tilting Table'
     'Ring Toss' 'Swinging Frames' 'Rope Swing Into Cargo Net'
     'Double Tilt Ladder' 'Crazy Cliffhanger' 'Downhill Pipe Drop'
     'Dancing Stones' 'Minefield' 'Cat Grab' 'Spikes Into Cargo Net'
     'Doorknob Arch' 'Piston Road' 'Silk Slider' 'Rope Jungle'
     'Butterfly Wall' 'Cannonball Incline' 'Propeller Bar' 'Snake Crossing'
     'Wind Chimes' 'Floating Monkey Bars' 'Invisible Ladder' 'Paddle Boards'
     'Tire Swing' 'Double Helix' 'Big Dipper' 'Floating Tiles'
     '(Modified) Ring Toss' 'Bungee Road' 'Flying Shelf Grab'
     'Mini Silk Slider' 'Spin Cycle' 'Hourglass Drop' 'Clear Climb'
     'Tilting Slider' 'Cargo Crossing' 'Swinging Spikes' 'Walking Bar'
     'Log Runner' 'I-Beam Crossing' 'Globe Grasper' 'Sonic Curve' 'Coin Flip'
     'Triple Swing' 'Roulette Row' 'Psycho Chainsaw' 'Area 51'
     'Floating Steps' 'Tick Tock' 'Escalator' 'Ring Jump' 'I-Beam Cross'
     'The Wedge' 'Helix Hang' 'Block Run' 'Pipe Fitter' 'The Clacker'
     'Fly Wheels' 'Disc Runner' 'Circuit Board' 'Ring Swing' 'Bar Hop'
     'Window Hang' 'Wall Drop' 'Rolling Thunder' 'Stair Hopper' 'Snake Run'
     'Giant Log Grip' 'Broken Bridge' 'Flying Squirrel' 'Giant Ring Swing'
     'Down Up Salmon Ladder' 'Wave Runner' 'Double Wedge' 'Wall Flip'
     'Keylock Hang' 'Curved Body Prop' 'Cannonball Drop' 'Battering Ram'
     'Swinging Peg Board' 'Elevator Climb' 'Sky Hooks' 'Spinball Wizard'
     'Rolling Pin' 'Wingnuts' 'Giant Cubes' 'Hang Glider' 'Broken Pipes'
     'Crank It Up' 'Iron Maiden' "Razor's Edge" 'I-Beam Gap' 'Nail Clipper'
     'Bouncing Spider' 'Rail Runner' 'Ninjago Roll' 'Double Dipper'
     'Parkour Run' 'Domino Pipes' 'Criss Cross Salmon Ladder' 'Swing Surfer'
     'Wingnut Alley' 'Peg Cloud' 'Time Bomb' 'Jumper Cables' 'Doorknob Drop'
     'Warped Wall / Mega Wall' 'Archer Steps' 'Baton Pass' 'Spider Trap'
     'Catch & Release' 'Tuning Forks' 'Fallout' 'Ring Turn' 'Slippery Summit'
     'Crazy Clocks' 'Wheel Flip' 'Spin Hopper' 'Cane Lane' 'Spinning Bowties'
     'Lightning Bolts' "Captain's Wheel" 'Double Twister' 'Diamond Dash'
     'The Hinge' 'Archer Alley' 'Jeep Run' 'Razor Beams' 'Twist & Fly'
     'Epic Catch & Release' 'Deja Vu' 'Water Walls' 'En Garde']
    
    
    Obstacle Order
    [ 1  2  3  4  5  6  7  8  9 10]
    
    


### Before I visualize the data to get a (literal) picture of the data, I'm going to do some indexing to play around.


```
amwDat.loc[(amwDat['Obstacle Name'] != 'Rope Swing') & (amwDat['Round/Stage'] == 'Qualifying') & (amwDat['Obstacle Order'] < 5)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Season</th>
      <th>Location</th>
      <th>Round/Stage</th>
      <th>Obstacle Name</th>
      <th>Obstacle Order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Quintuple Steps</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Rolling Barrel</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Jumping Spider</td>
      <td>4</td>
    </tr>
    <tr>
      <th>40</th>
      <td>2</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Quad Steps</td>
      <td>1</td>
    </tr>
    <tr>
      <th>42</th>
      <td>2</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Bridge of Blades</td>
      <td>3</td>
    </tr>
    <tr>
      <th>43</th>
      <td>2</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Jumping Spider</td>
      <td>4</td>
    </tr>
    <tr>
      <th>78</th>
      <td>3</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Quad Steps</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>3</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Log Grip</td>
      <td>2</td>
    </tr>
    <tr>
      <th>80</th>
      <td>3</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Bridge of Blades</td>
      <td>3</td>
    </tr>
    <tr>
      <th>81</th>
      <td>3</td>
      <td>Venice</td>
      <td>Qualifying</td>
      <td>Jump Hang</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



This is a very random random of indexing, but I found it interesting. Notice how after I've limited Obstacle Name to everything but 'Rope Swing', Round/Stage to 'Qualifying', and making Obstacle Order less than 5, there are only 10 entries left.

## I need to also check for any missing values in the data, because those dudes can be very pesky.


```
def num_missing(x):
  return(sum(x.isnull()))

# This function will help check each column in the dataframe to see if there are any missing values

amwDat.apply(num_missing, axis=0)
```




    Season            0
    Location          0
    Round/Stage       0
    Obstacle Name     0
    Obstacle Order    0
    dtype: int64



Awesome! Every single column has zero missing values! This makes our job much easier!


```
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
```

# Time to do some visualizations!


```
amwDat.plot.scatter(x='Season', y='Obstacle Order')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f8a668032b0>




![png](American%20Ninja%20Warrior%20Obstacle%20Course%20Project_files/American%20Ninja%20Warrior%20Obstacle%20Course%20Project_26_1.png)


## It only takes a few seconds to really just how silly this visualization is. Most of the data in this set is categorical and not numerical, and even those with numeric values are really just ordinal  data, meaning that they have a scale of low to high, but besides that, they don't have much numeric value.

# Now let's see which values appear most often for each column.


```
for column in colNames:
  print(column)
  print(amwDat[column].mode())
  print('\n')
```

    Season
    0     7
    1     9
    2    10
    dtype: int64
    
    
    Location
    0    Las Vegas
    dtype: object
    
    
    Round/Stage
    0    Finals (Regional/City)
    dtype: object
    
    
    Obstacle Name
    0    Warped Wall
    dtype: object
    
    
    Obstacle Order
    0    1
    dtype: int64
    
    


## From this we can see that the most common location in the dataset is Las Vegas, the most common round/stage listed in the dataset is the regional/city finals, and the most common obstacle is the warped wall.
