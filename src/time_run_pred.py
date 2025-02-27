import time
import matplotlib.pyplot as plt
from myprogram import MyModel

def measure_run_pred(model, test_data):
    start_time = time.time()
    model.run_pred(test_data)
    end_time = time.time()
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
    "This input is the longest one yet and should be exactly two hundred characters in length, designed to push the boundaries of system handling, making sure that inputs at the max length are still valid."
]

    work_dir = '../work'
    model = MyModel.load(work_dir)

    input_lengths = [len(data) for data in test_data]
    times = [measure_run_pred(model, [data]) for data in test_data]

    plt.figure(figsize=(10, 6))
    plt.scatter(input_lengths, times)
    plt.xlabel('Num characters')
    plt.ylabel('Seconds')
    plt.title('Time vs Number of Characters')
    plt.show()

if __name__ == '__main__':
    main()