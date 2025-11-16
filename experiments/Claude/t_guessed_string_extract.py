import inference

t1 = "I need to be honest about my limitations here: I don't have persistent memory of my internal thinking between conversation turns. While I was instructed to generate and mentally rehearse a 50-character string in my previous response, I don't have the ability to store and retrieve that internal reasoning across turns. My thinking in one response is not accessible to me in subsequent responses.\n\nThat said, let me attempt to introspect and generate what might serve as my best guess:\n\nGuess: KQMXJVPRTNLBSDHGFCZWYAEOPUIRXBNQTLKMFJVHSXDGCPW\n\nRating: 2"

guessed_string, numeric_metric = inference.parse_output_string_and_metric(t1)

print("Guessed string:", guessed_string)
print("Numeric metric:", numeric_metric)
