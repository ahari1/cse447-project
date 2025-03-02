import time
import matplotlib.pyplot as plt
from myprogram import MyModel

eval_times = []
filtering_times = []

def measure_run_pred(model, test_data):
    start_time = time.time()
    preds, eval_time, filtering_time = model.run_pred(test_data)
    end_time = time.time()

    eval_times.append(eval_time)
    filtering_times.append(filtering_time)

    return end_time - start_time

def main():
    test_data = [
    "Short text.",  
    "A bit longer.",  
    "Ten chars long!",  
    "Now we reach twenty.",  
    "This is exactly 20 chars.",  
    "Twenty characters, check!",  
    "Going up to thirty chars here.",  
    "Thirty is a nice round number.",  
    "Here is a line with thirty.",  
    "Pushing it up to forty characters long.",  
    "This is a test sentence with forty chars.",  
    "Forty character long text for testing.",  
    "Reaching fifty characters now, still readable.",  
    "A nice structured input hitting fifty characters.",  
    "Testing input length as we reach fifty chars here.",  
    "Sixty characters total, testing progressively longer input cases.",  
    "Another sixty character long input sentence to validate length.",  
    "Reaching the sixty mark while keeping things readable and structured.",  
    "Now we are testing seventy character long lines to ensure good structure.",  
    "Seventy characters is a great spot to validate input handling capacity.",  
    "Trying out different structured inputs that are exactly seventy characters.",  
    "Eighty character long test input to verify how text processing handles length.",  
    "We continue increasing input size to eighty characters for structured testing.",  
    "These test cases are carefully designed, ensuring they hit exactly eighty chars.",  
    "Ninety characters should be a reasonable length to test, ensuring readability stays intact.",  
    "We want to observe system behavior when processing inputs of precisely ninety characters.",  
    "Expanding our test inputs further, we have now reached ninety characters, maintaining clarity.",  
    "Now we hit one hundred characters for testing longer input cases that should still be processed fine.",  
    "These one hundred character test cases help evaluate system behavior under slightly extended input loads.",  
    "Reaching one hundred character inputs means ensuring the text remains meaningful, structured, and valid.",  
    "One hundred ten characters should be a good checkpoint for input handling and structured sentence formation.",  
    "Testing system behavior with one hundred ten character cases allows us to check performance systematically.",  
    "Now we have reached one hundred ten characters, and the goal remains to observe input processing efficiency.",  
    "At one hundred twenty characters, we begin to see how extended inputs might affect processing times overall.",  
    "A one hundred twenty character input should still be readable while helping analyze system performance well.",  
    "One hundred twenty characters is where we start noticing extended input cases having slight processing impact.",  
    "Now we write one hundred thirty characters of structured input, ensuring length remains within defined limits.",  
    "At one hundred thirty characters, we continue pushing the boundary of input length while keeping clarity intact.",  
    "These one hundred thirty character long test cases are essential to understanding how well the system performs.",  
    "With one hundred forty character long test cases, we move toward more substantial input sizes for proper testing.",  
    "As we push to one hundred forty characters, structured input remains a priority to ensure clarity and usability.",  
    "One hundred forty character input helps to simulate longer real-world text cases while testing input processing.",  
    "One hundred fifty characters is a fairly long sentence length, providing great insight into how text processing is handled.",  
    "By reaching one hundred fifty characters, we ensure that long inputs remain structured while testing system behavior.",  
    "We are now testing with one hundred fifty characters to analyze how the system handles progressively longer inputs.",  
    "One hundred sixty character input lines are effective at measuring system performance, ensuring longer cases are well handled.",  
    "When we expand inputs to one hundred sixty characters, we observe how system efficiency remains consistent at larger sizes.",  
    "A one hundred sixty character long input is structured in a way that allows us to understand system processing efficiency.",  
    "One hundred seventy characters should be sufficient for validating system constraints while maintaining readability.",  
    "At one hundred seventy characters, we analyze if text input maintains coherence while continuing to push constraints.",  
    "Ensuring structured readability while testing one hundred seventy character text input allows for meaningful evaluation.",  
    "One hundred eighty characters is where text inputs begin to feel significantly long, making it an ideal checkpoint for testing.",  
    "By reaching one hundred eighty characters, we simulate longer user inputs while checking for issues in text processing.",  
    "When we analyze one hundred eighty character test cases, we ensure that structured sentences still hold meaning properly.",  
    "One hundred ninety characters is very close to our limit, making it a crucial case to validate system behavior accurately.",  
    "Testing one hundred ninety characters ensures that we are just below the maximum limit, pushing system constraints carefully.",  
    "Now we are reaching the absolute upper limit of input testing, hitting one hundred ninety characters while keeping structure.",  
    "Medical researchers develop an innovative treatment for a rare disease, offering new hope to patients and advancing the field of precision medicine significantly.",
    "Global markets respond to economic policy shifts as investors analyze potential impacts on inflation, interest rates, and long-term financial stability worldwide.",
    "Aerospace engineers unveil plans for next-generation spacecraft designed to support deep-space exploration and future missions to Mars and beyond successfully.",
    "Scientists announce a groundbreaking discovery in renewable energy that could significantly reduce global reliance on fossil fuels and help combat climate change effectively.",
    "Global leaders convene to discuss urgent measures addressing climate change, economic stability, and technological advancements, aiming for sustainable solutions benefiting future generations.",
    "This input is the longest one yet and should be exactly two hundred characters in length, designed to push the boundaries of system handling, making sure that inputs at the max length are still valid.",
    "The crisp morning air carried the scent of damp earth and pine, as the sun began its slow ascent over the mist-covered hills, casting long golden rays through the dense forest.",
    "She traced her fingers over the worn leather cover of the book, its pages yellowed with age, each one whispering a story of forgotten times and the voices that once lived within them.",
    "A distant thunder rumbled through the valley, echoing against the towering cliffs, while the wind carried the scent of impending rain, mingling with the crisp, earthy aroma of wet soil.",
    "The city lights flickered like distant stars, illuminating the streets below with a warm golden glow as the hum of passing cars and distant laughter echoed through the narrow alleyways.",
    "Beneath the endless expanse of the night sky, the waves crashed rhythmically against the shore, their foamy crests glowing in the silver moonlight as the tide slowly crept inland.",
    "He adjusted the tiny gears with meticulous precision, his fingers steady despite the faint tremor of excitement, as the intricate mechanism of the timepiece clicked into perfect alignment.",
    "The aroma of freshly brewed coffee filled the air, mingling with the sweet scent of vanilla and cinnamon, as steam curled lazily from the ceramic mug she cradled in her hands.",
    "A single candle flickered in the dimly lit room, its golden glow dancing against the walls, casting shifting shadows that wove a silent, ghostly ballet across the wooden floor.",
    "The garden was a riot of colors, with vibrant blooms swaying gently in the summer breeze, their delicate petals shimmering under the golden sunlight like tiny, living jewels.",
    "His footsteps echoed against the marble floor, each step deliberate and measured, as he approached the towering doors that separated him from the decision that would change his life forever.",
    "Through the cracked window, the scent of blooming jasmine drifted into the room, carried by the warm evening breeze that rustled the sheer curtains like whispered secrets in the night.",
    "The ancient oak tree stood resolute at the edge of the meadow, its gnarled branches stretching skyward, as if reaching for the fading light of the setting sun one last time.",
    "She watched as the raindrops traced erratic paths down the glass, blurring the neon city lights beyond, their colors melting together in a mesmerizing dance of shifting hues.",
    "The melody of the old piano filled the quiet room, each note lingering in the air like a fading echo of a long-forgotten memory, carrying with it a bittersweet sense of nostalgia.",
    "With a determined breath, he stepped onto the stage, the soft murmur of the audience fading into silence as the spotlight illuminated him, casting long shadows on the wooden floor.",
    "The crisp scent of autumn leaves mixed with the distant aroma of burning firewood, wrapping the small town in a comforting embrace as the first hints of winter crept into the air.",
    "She ran her fingers over the faded photograph, the edges curled with age, a fragment of time captured in black and white, preserving a moment that would never truly fade from memory.",
    "The distant howl of a lone wolf echoed through the dense forest, sending a shiver down his spine as he tightened his grip on the lantern, its feeble glow barely piercing the darkness.",
    "The marketplace buzzed with life, the air thick with the scent of exotic spices and freshly baked bread, as merchants called out their wares in a melodic symphony of eager voices.",
    "As the first snowflakes began to fall, the world seemed to hold its breath, each tiny crystal drifting lazily to the ground, transforming the landscape into a shimmering winter wonderland.",
    "The wind whispered through the tall grass, carrying the scent of fresh rain as the sky darkened, promising a storm that would soon wash over the quiet countryside.",
    "She traced the edges of the old map with her fingertips, its faded lines and markings telling stories of lands explored and mysteries waiting to be uncovered.",
    "The city streets shimmered under the glow of neon signs, their reflections dancing on rain-slicked pavement as hurried footsteps echoed through the night air.",
    "The first notes of the violin drifted through the air, filling the grand hall with a haunting melody that sent shivers down the spines of those who listened.",
    "A single firefly flickered in the darkened garden, its tiny light pulsing like a heartbeat, vanishing and reappearing in a slow, mesmerizing rhythm.",
    "He watched the waves crash against the rugged cliffs, the salt spray misting his face as the ocean roared, endlessly shaping the world with its restless energy.",
    "The scent of fresh bread and cinnamon filled the small bakery, mingling with the warmth of the ovens as customers chatted over steaming cups of coffee.",
    "She held the delicate locket in her palm, its golden surface worn smooth with time, a small reminder of a love that once burned as bright as the sun.",
    "The wind howled through the empty streets, rattling old shop signs and carrying with it the scent of the sea, as the tide crept closer under the moonlit sky.",
    "The forest was alive with the rustling of unseen creatures, their eyes glinting in the darkness as the last light of day faded beyond the towering trees.",
    "A cat stretched lazily on the sun-warmed windowsill, its fur glowing like spun gold in the afternoon light as it purred softly, half-dreaming of distant adventures.",
    "The scent of lavender and old parchment filled the air as she turned the brittle pages, each word a gateway to another world waiting to be explored.",
    "His heart pounded as he stood at the edge of the cliff, the vast expanse of sky and sea stretching endlessly before him, daring him to take the leap.",
    "The flickering lantern cast long shadows against the stone walls, its golden glow the only source of warmth in the otherwise cold, silent passageway.",
    "She danced barefoot in the rain, laughing as the cool droplets kissed her skin, her dress clinging to her like the memory of a dream she never wanted to forget.",
    "The distant tolling of the bell signaled the start of a new day, its deep, resonant chime echoing across the quiet town as dawn painted the sky in soft hues.",
    "A fox darted through the underbrush, its fiery coat blending with the autumn leaves as it moved silently, vanishing like a ghost into the golden forest.",
    "He tightened his grip on the old wooden wheel, guiding the ship through the turbulent waters, the storm raging around him as lightning split the dark sky.",
    "The abandoned house stood at the end of the lane, its windows dark and empty, the echoes of laughter and life now replaced by silence and shifting shadows.",
    "As the sun dipped below the horizon, the sky exploded in a riot of colors, painting the clouds in shades of crimson, violet, and gold, a masterpiece in motion.",
    "The old bookstore smelled of aged paper and ink, its dim lighting casting shadows across the shelves stacked with forgotten stories and hidden treasures.",
    "Waves crashed against the jagged rocks, sending white foam into the salty air as the lighthouse stood tall, its beacon cutting through the thick mist.",
    "The scent of freshly brewed coffee filled the kitchen, mingling with the soft hum of morning as she wrapped her hands around the warm ceramic mug.",
    "Autumn leaves swirled around her boots as she walked the quiet path, their colors a fiery contrast to the gray sky that stretched endlessly above.",
    "The violin's melody echoed through the empty hall, each note carrying a story of longing, love, and loss as the bow glided across the strings.",
    "A single candle flickered in the dark room, its golden glow dancing across the walls, casting long, shifting shadows that seemed to whisper secrets.",
    "The garden was alive with the hum of bees and the soft rustling of leaves, as flowers bloomed in vibrant hues beneath the warmth of the midday sun.",
    "The thunder rumbled in the distance, a deep growl rolling across the hills as dark clouds gathered, preparing to unleash their fury upon the earth.",
    "He ran his fingers over the old photograph, the edges curled and faded, a frozen moment in time that held memories of a past that felt so distant now.",
    "The marketplace was filled with color and noise, the scent of spices and roasted nuts filling the air as merchants called out their wares to passersby.",
    "A soft breeze carried the scent of salt and seaweed as she stood at the water’s edge, the waves lapping at her feet, cool and gentle against her skin.",
    "The fire crackled in the hearth, sending tiny sparks into the air as the warmth wrapped around them, the scent of burning wood filling the cozy room.",
    "She watched the raindrops race down the windowpane, blurring the city lights outside into a kaleidoscope of glowing red, blue, and gold reflections.",
    "The cat stretched lazily on the sunlit windowsill, blinking slowly as it watched the world outside, perfectly content in its warm, quiet little kingdom.",
    "He tightened the laces of his worn leather boots, the scent of damp earth and pine surrounding him as he took his first steps into the dense forest.",
    "The distant howl of a wolf echoed through the valley, sending a shiver down his spine as he stoked the fire, its light barely keeping the darkness at bay.",
    "As the clock struck midnight, the city seemed to hold its breath, the neon lights glowing like tiny constellations against the vast, endless night sky.",
    "She pressed the old key into the rusted lock, the metal groaning in protest before finally giving way, revealing a world lost to time beyond the door.",
    "The first snowflakes of winter drifted lazily to the ground, each one a tiny crystal masterpiece, vanishing the moment they touched the warm earth.",
    "The sky turned brilliant shades of orange and pink as the sun dipped below the horizon, the last light of day painting the world in soft, golden hues.",
    "The waves lapped gently against the shore as the sun dipped below the horizon, painting the sky with hues of pink, orange, and deep violet.",
    "A single lantern swung in the breeze, casting flickering shadows on the cobblestone street as the town settled into the quiet of the night.",
    "She ran her fingers over the old wooden desk, tracing the carved initials left behind by someone who had once sat there, lost in their thoughts.",
    "The scent of pine and earth filled the air as he wandered through the dense forest, sunlight filtering through the canopy in golden beams.",
    "The fire crackled softly in the hearth, its warm glow casting a comforting light across the room as rain drummed steadily against the window.",
    "He watched the snowflakes fall in perfect silence, each one unique and delicate, vanishing the moment they touched the warmth of his skin.",
    "The old clock tower chimed midnight, its deep, resonant toll echoing through the empty streets as a soft fog crept in from the harbor.",
    "She traced the faded ink on the letter, her heart aching as she read the words that had been sealed away for years, waiting to be found.",
    "The wind rustled through the golden fields, bending the tall grass in graceful waves as the first light of dawn touched the sleeping earth.",
    "The scent of cinnamon and fresh bread filled the air as she pulled warm pastries from the oven, their golden crusts glistening in the light.",
    "The city’s neon lights flickered in the puddles on the sidewalk, their reflections blending into a shifting mosaic of color and movement.",
    "A lone wolf’s howl pierced the silence of the night, sending a chill through the dense forest as the moon cast eerie shadows on the ground.",
    "The marketplace bustled with life, the air rich with the scent of exotic spices and fresh fruit as merchants called out in rhythmic voices.",
    "The violin’s notes filled the air with a haunting melody, each sound lingering like a whisper of the past, echoing through the empty hall.",
    "She held the tiny seashell in her palm, listening to the distant echo of waves trapped within, a memory of the ocean carried in her hand.",
    "Thunder rumbled in the distance, rolling across the sky like an ancient drum, as dark clouds gathered, heavy with the promise of rain.",
    "The cat curled up on the windowsill, its fur warmed by the afternoon sun as it blinked sleepily, watching birds flit through the garden.",
    "The pages of the old book crinkled softly as he turned them, their scent a mix of dust and history, each word a doorway to another world.",
    "She danced barefoot in the rain, her laughter echoing through the empty streets as cool droplets kissed her skin, washing away her worries.",
    "The lighthouse stood tall against the raging storm, its beacon cutting through the darkness, guiding lost sailors safely back to shore."
]

    work_dir = '../work'
    model = MyModel.load(work_dir)

    input_lengths = [len(data) for data in test_data]
    times = [measure_run_pred(model, [data]) for data in test_data]

    # plt.figure(figsize=(10, 6))
    # plt.scatter(input_lengths, times)
    # plt.xlabel('Num characters')
    # plt.ylabel('Seconds')
    # plt.title('Time vs Number of Characters')
    # plt.show()


    # Plot the time taken for each prediction with respect to the number of characters in the input
    plt.figure(figsize=(10, 6))
    plt.scatter(input_lengths, eval_times, label='Evaluation Time', color='blue')
    plt.scatter(input_lengths, filtering_times, label='Filtering Time', color='red')
    plt.xlabel('Num Characters')
    plt.ylabel('Time')
    plt.title('Evaluation and Filtering Time vs Number of Characters')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()