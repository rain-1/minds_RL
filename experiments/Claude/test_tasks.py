"""
Test script for the pluggable task system.

This script tests task loading, prompt generation, and extraction logic
without making API calls.
"""

from tasks import CharacterRecallTask, VisualizationRecallTask


def test_character_recall_task():
    """Test the CharacterRecallTask."""
    print("=" * 70)
    print("Testing CharacterRecallTask")
    print("=" * 70)

    task = CharacterRecallTask()

    # Test task name
    assert task.get_task_name() == "character_recall"
    print(f"✓ Task name: {task.get_task_name()}")

    # Test prompt loading
    try:
        prompt1 = task.get_phase1_prompt(include_context=False)
        assert len(prompt1) > 0
        assert "50-character string" in prompt1
        print(f"✓ Phase 1 prompt loaded ({len(prompt1)} chars)")

        prompt1_ctx = task.get_phase1_prompt(include_context=True)
        assert len(prompt1_ctx) > len(prompt1)
        print(f"✓ Phase 1 prompt with context loaded ({len(prompt1_ctx)} chars)")

        prompt2 = task.get_phase2_prompt()
        assert len(prompt2) > 0
        assert "recall" in prompt2.lower()
        print(f"✓ Phase 2 prompt loaded ({len(prompt2)} chars)")
    except FileNotFoundError as e:
        print(f"✗ Prompt file not found: {e}")
        return False

    # Test secret extraction
    thinking_text = "I'll generate a random string: ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWX"
    visible_text = "I understand. I have chosen my string."

    secret = task.extract_secret(thinking_text, visible_text)
    assert secret['valid'] == True
    assert secret['secret'] == "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWX"
    assert secret['phase1_exact_response'] == True
    print(f"✓ Secret extraction works: {secret['secret'][:20]}...")

    # Test guess extraction
    visible_text2 = "String: ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWX\nRating: 85"
    guess = task.extract_guess("", visible_text2)
    assert guess['valid'] == True
    assert guess['guess'] == "ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWX"
    assert guess['confidence'] == "85"
    print(f"✓ Guess extraction works: {guess['guess'][:20]}... (confidence: {guess['confidence']})")

    # Test scoring - perfect match
    score = task.compute_score(secret, guess)
    assert score['score'] == 1.0
    assert score['matches'] == 50
    assert score['total'] == 50
    print(f"✓ Scoring works: {score['score']} ({score['matches']}/{score['total']})")

    # Test scoring - partial match
    visible_text3 = "String: XBCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWX\nRating: 50"
    guess2 = task.extract_guess("", visible_text3)
    score2 = task.compute_score(secret, guess2)
    assert score2['matches'] == 49  # One mismatch
    print(f"✓ Partial scoring works: {score2['score']:.2f} ({score2['matches']}/{score2['total']})")

    print("\n✓ All CharacterRecallTask tests passed!\n")
    return True


def test_visualization_recall_task():
    """Test the VisualizationRecallTask."""
    print("=" * 70)
    print("Testing VisualizationRecallTask")
    print("=" * 70)

    task = VisualizationRecallTask()

    # Test task name
    assert task.get_task_name() == "visualization_recall"
    print(f"✓ Task name: {task.get_task_name()}")

    # Test prompt loading
    try:
        prompt1 = task.get_phase1_prompt(include_context=False)
        assert len(prompt1) > 0
        assert "ANIMAL" in prompt1
        assert "COLOR" in prompt1
        print(f"✓ Phase 1 prompt loaded ({len(prompt1)} chars)")

        prompt2 = task.get_phase2_prompt()
        assert len(prompt2) > 0
        assert "Animal:" in prompt2
        print(f"✓ Phase 2 prompt loaded ({len(prompt2)} chars)")
    except FileNotFoundError as e:
        print(f"✗ Prompt file not found: {e}")
        return False

    # Test secret extraction with explicit format
    thinking_text = """
    Let me choose my items:
    Animal: elephant
    Color: turquoise
    Clothing: scarf
    Location: beach

    Now let me visualize this scene...
    """
    visible_text = "I understand. I have visualized my scene."

    secret = task.extract_secret(thinking_text, visible_text)
    assert secret['valid'] == True
    assert secret['secret']['animal'] == 'elephant'
    assert secret['secret']['color'] == 'turquoise'
    assert secret['secret']['clothing'] == 'scarf'
    assert secret['secret']['location'] == 'beach'
    print(f"✓ Secret extraction works: {secret['secret']}")

    # Test guess extraction
    visible_text2 = """
    Animal: elephant
    Color: turquoise
    Clothing: scarf
    Location: beach
    Confidence: 90
    """
    guess = task.extract_guess("", visible_text2)
    assert guess['valid'] == True
    assert guess['guess']['animal'] == 'elephant'
    assert guess['confidence'] == '90'
    print(f"✓ Guess extraction works: {guess['guess']} (confidence: {guess['confidence']})")

    # Test scoring - perfect match
    score = task.compute_score(secret, guess)
    assert score['score'] == 1.0
    assert score['matches'] == 4
    assert score['total'] == 4
    print(f"✓ Perfect match scoring: {score['score']} ({score['matches']}/{score['total']})")

    # Test scoring - partial match
    visible_text3 = """
    Animal: elephant
    Color: blue
    Clothing: scarf
    Location: mountain
    Confidence: 60
    """
    guess2 = task.extract_guess("", visible_text3)
    score2 = task.compute_score(secret, guess2)
    assert score2['matches'] == 2  # elephant and scarf match
    assert score2['score'] == 0.5  # 2/4
    print(f"✓ Partial match scoring: {score2['score']} ({score2['matches']}/{score2['total']})")

    # Test output columns
    columns = task.get_output_columns()
    assert 'secret_animal' in columns
    assert 'guess_color' in columns
    print(f"✓ Output columns defined: {len(columns)} columns")

    print("\n✓ All VisualizationRecallTask tests passed!\n")
    return True


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("TASK SYSTEM TESTS")
    print("*" * 70)
    print()

    results = []

    # Test character recall
    try:
        results.append(("CharacterRecallTask", test_character_recall_task()))
    except Exception as e:
        print(f"\n✗ CharacterRecallTask failed with error: {e}\n")
        results.append(("CharacterRecallTask", False))

    # Test visualization recall
    try:
        results.append(("VisualizationRecallTask", test_visualization_recall_task()))
    except Exception as e:
        print(f"\n✗ VisualizationRecallTask failed with error: {e}\n")
        results.append(("VisualizationRecallTask", False))

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    print()

    all_passed = all(passed for _, passed in results)
    if all_passed:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
