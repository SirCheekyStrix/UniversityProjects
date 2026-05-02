#ifndef SMOK_H
#define SMOK_H

#include <string>
#include "Owca.h"
#include "OwcaNadziana.h"

class Smok {
    public:
        enum Rasa {
            OGNIOSMOK,
            WODNYSMOK,
            ZIEMNYSMOK
        };

        Smok(Rasa rasa, int limitSiarki, int limitWody);
        void zionie_ogniem();
        void zjedzOwce(Owca* owca);
        void pijWode(int woda);
        std::string getRasa() const;
        bool czyZywy() const;

    private:
        Rasa rasa;
        int zjedzonaSiarka;
        int limitSiarki;
        int poziomPragnienia;
        int limitWody;
        bool zywy;
};

#endif

