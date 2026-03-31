from collections import Counter

def vote(predictions: dict) -> str:

    answers = list(predictions.values())
    vote_count = Counter(answers)

    print("\n" + "-" * 50)
    print("         Voting Breakdown")
    print("-" * 50)

    for plant_type, count in vote_count.most_common():
        bar = "█" * count
        print(f"  {plant_type:<15} {bar}  ({count} vote(s))")

    print("-" * 50)

    top_count = vote_count.most_common(1)[0][1]
    tied_winners = [k for k, v in vote_count.items() if v == top_count]

    if len(tied_winners) > 1:
        print(f"\n  Tie detected between: {tied_winners}")
        print(f"  Picking first: {tied_winners[0]}")
        return tied_winners[0]

    winner = vote_count.most_common(1)[0][0]
    winner_votes = vote_count.most_common(1)[0][1]
    print(f"\n  Winner: {winner} with {winner_votes} out of 5 votes")
    return winner

