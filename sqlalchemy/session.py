import pandas as pd
import sqlalchemy
from sqlalchemy import Column, String, Integer, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import settings


dbcon = sqlalchemy.engine_from_config(
    settings.ENGINE_URL, prefix="BASKETBALL_NBA_TEST."
)
df = pd.read_sql("select * from Input_Test", dbcon)

modelcon = sqlalchemy.engine_from_config(
    settings.ENGINE_URL, prefix="BASKETBALL_NBA_MODEL_TEST."
)
Base = declarative_base()

class Output_Test(Base):
    __tablename__ = "Output_Test"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    outname = Column(String)

    def __repr__(self):
        return f"Name is {self.outname}"

Base.metadata.create_all(modelcon)
Session = sessionmaker(bind=modelcon)
sess = Session()

sess.query(Output_Test).get(3)
sess.query(Output_Test).filter(Output_Test.outname.in_(["a", "b"])).count()

tmp = sess.query(Output_Test).filter_by(outname='a').all()
sess.delete(tmp)
# sess.commit()

sess.query(Output_Test).filter_by(outname='a').delete()
sess.commit()
sess.query(Output_Test).filter(and_(Output_Test.outname=='a', Output_Test.id==1)).delete()
sess.query(Output_Test).all()
