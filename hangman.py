def main():
    mistakes = 8

    print("Welcome to hangman for two players.\nPlayer 1 selects a word and Player 2 may only make "+str(mistakes)+" mistakes in attempting to guess characters.")
    word = input("Player 1, type a word: ").lower()
    print(("#"*8+"\n")*8+"#"*8)

    progress = []
    for i in range (len(word)):
        progress.append("0")
    
    while mistakes > 0:
        display = ""
        for i in range(len(word)):
            if progress[i] == "0":
                display += "_ "
            else:
                display += progress[i] + " "
        print(display)

        print(str(abs(mistakes-8))+" mistake(s) made so far.")
        attempt = ""
        while attempt is "":
            attempt = input("Player 2, make a guess: ").lower()
            if len(attempt) > 1:
                attempt = ""
        

        for i in range(len(word)):
            if attempt == word[i]:
                progress[i] = attempt
                if wincheck(progress, word):
                    mistakes = 0
                    print("\nPlayer 2 guessed the word!\nThe word was: "+word)
        
        if attempt in word and not wincheck(progress, word):
            print("Player 2 guessed right!\n")
        else:
            if mistakes == 1:
                print("\nPlayer 2 has failed to guess the word!")
                break
            else:
                if mistakes != 0:
                    print("Player 2 guessed wrong!\n")
                    mistakes -= 1
        
def wincheck(progress, word):
    comp = ""
    for i in range (len(word)):
        comp += progress[i]
    return comp == word
    
main()
